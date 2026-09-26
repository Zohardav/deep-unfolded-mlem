"""Run the complete, intentionally compact reproduction from frozen data."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from data_and_operators import (
    COMPTON_BAG_EVENTS, EPS, ComptonRowCache, ConvolutionOperator, build_compton_cache, gaussian_psf, load_rl,
    make_compton_manifest, point_targets, rectangle_event_sets, rectangle_target,
    regularizer_terms,
)
from models import DeepUnfoldedMLEM, conventional_reconstruct, hybrid_loss, nmse, uniform_image
from plotting import figure1, figure3, nmse_figure, runtime_tables


ROOT = Path(__file__).resolve().parent
CONFIG = {
    "seed": 20260920,
    "layers": 10,
    "rl": {"batch": 64, "epochs_seq": 6, "epochs_e2e": 6, "lr": 2e-3,
           "gamma_grid": [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0], "iterations": 300},
    "compton": {"batch": 32, "epochs_seq": 3, "epochs_e2e": 3, "lr": 1e-3,
                "gamma_grid": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], "iterations": 100,
                "mu_t_per_mm": 0.035, "rectangle_realizations": 30,
                "bag_events": COMPTON_BAG_EVENTS},
    "scheduler": {"factor": 0.5, "patience": 2},
    "early_stopping_patience": 20,
    "lambda_shape": 1.0, "lambda_loc": 0.10, "lambda_data": 0.01,
}


def seed_all(seed: int) -> None:
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def batches(n: int, batch: int, *, shuffle: bool, seed: int, limit: int | None = None):
    order = np.arange(min(n, limit or n))
    if shuffle: np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch): yield order[start:start + batch]


def adjoint_error(operator, x: torch.Tensor, z: torch.Tensor) -> float:
    lhs = (operator.forward(x) * z).sum(); rhs = (x * operator.adjoint(z)).sum()
    return float((lhs - rhs).abs() / torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1e-12))


@torch.no_grad()
def tune_rl_gamma(op, validation, device, cfg, quick: bool) -> tuple[float, list[dict]]:
    rows = []
    limit = 128 if quick else len(validation["truth"]); iterations = 30 if quick else 300
    for gamma in cfg["gamma_grid"]:
        values = []
        for ids in batches(limit, 128, shuffle=False, seed=0):
            y = validation["counts"][ids].to(device) / 5000.0; target = validation["truth"][ids].to(device)
            pred = conventional_reconstruct(op, y, (32, 32), iterations, gamma, keep_trajectory=False)[-1]
            values.append(nmse(pred, target).cpu().numpy())
        v = np.concatenate(values); rows.append({"gamma": gamma, "validation_nmse": float(v.mean())})
        print(f"RL gamma {gamma:g}: validation NMSE {v.mean():.6f}", flush=True)
    return float(min(rows, key=lambda r: r["validation_nmse"])["gamma"]), rows


@torch.no_grad()
def tune_compton_gamma(cache, manifest, device, cfg, quick: bool) -> tuple[float, list[dict]]:
    bags0, sources0 = manifest["validation_event_indices"], manifest["validation_source_ids"]
    limit = 128 if quick else len(bags0); iterations = 20 if quick else 100; rows = []
    for gamma in cfg["gamma_grid"]:
        values = []
        for ids in batches(limit, 64, shuffle=False, seed=0):
            ev = bags0[ids]; op = cache.operator(ev, device); y = torch.ones(len(ids), bags0.shape[1], device=device)
            target = point_targets(sources0[ids], device)
            pred = conventional_reconstruct(op, y, (81, 81), iterations, gamma, keep_trajectory=False)[-1]
            values.append(nmse(pred, target).cpu().numpy())
        v = np.concatenate(values); rows.append({"gamma": gamma, "validation_nmse": float(v.mean())})
        print(f"Compton gamma {gamma:g}: validation NMSE {v.mean():.6f}", flush=True)
    return float(min(rows, key=lambda r: r["validation_nmse"])["gamma"]), rows


def _run_epoch(model, depth, provider, optimizer=None) -> dict[str, float]:
    training = optimizer is not None; model.train(training)
    sums = {k: 0.0 for k in ("total", "shape", "location", "data")}; count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in provider:
            op, y, target, observed, xx, yy, scale = batch
            if training: optimizer.zero_grad(set_to_none=True)
            pred = model(op, y, target.shape[-2:], depth=depth)[-1]
            loss, parts = hybrid_loss(pred, target, op, observed, model.layers[depth - 1], xx, yy, data_scale=scale)
            if training:
                loss.backward(); torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0); optimizer.step()
            n = len(target); count += n; sums["total"] += float(loss.detach()) * n
            for k, v in parts.items(): sums[k] += float(v.detach()) * n
    return {k: v / count for k, v in sums.items()}


def train_interleaved(model, train_provider, val_provider, cfg: dict, checkpoint_dir: Path, *, quick: bool) -> list[dict]:
    """Seq L1; Seq L2/E2E L1:L2; ...; Seq L10/E2E L1:L10."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stages = [("seq", k) for k in range(1, 11)]
    stages = [("seq", 1)] + [item for k in range(2, 11) for item in (("seq", k), ("e2e", k))]
    history: list[dict] = []
    existing = sorted(checkpoint_dir.glob("stage_??.pt"))
    start = 0
    if existing:
        state = torch.load(existing[-1], map_location=next(model.parameters()).device, weights_only=False)
        model.load_state_dict(state["model"]); history = state["history"]; start = state["stage_index"] + 1
        print(f"Resuming after stage {start}/{len(stages)}", flush=True)
    for stage_index, (kind, depth) in enumerate(stages[start:], start=start):
        for p in model.parameters(): p.requires_grad_(False)
        active = [model.layers[depth - 1]] if kind == "seq" else model.layers[:depth]
        for layer in active:
            for p in layer.parameters(): p.requires_grad_(True)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=cfg["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=2)
        epochs = 1 if quick else cfg["epochs_seq" if kind == "seq" else "epochs_e2e"]
        best, best_state, stale = float("inf"), None, 0
        print(f"Stage {stage_index + 1:02d}/19: {kind.upper()} depth {depth}", flush=True)
        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(model, depth, train_provider(stage_index, epoch, True), optimizer)
            val_metrics = _run_epoch(model, depth, val_provider(stage_index, epoch, False))
            scheduler.step(val_metrics["total"])
            row = {"stage": stage_index + 1, "kind": kind, "depth": depth, "epoch": epoch,
                   "lr": optimizer.param_groups[0]["lr"],
                   **{f"train_{k}": v for k, v in train_metrics.items()},
                   **{f"validation_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_metrics["total"] < best:
                best = val_metrics["total"]; best_state = copy.deepcopy(model.state_dict()); stale = 0
            else: stale += 1
            if stale >= CONFIG["early_stopping_patience"]: break
        if best_state is None: raise RuntimeError("stage produced no checkpoint")
        model.load_state_dict(best_state)
        torch.save({"model": model.state_dict(), "history": history, "stage_index": stage_index,
                    "kind": kind, "depth": depth, "best_validation_loss": best}, checkpoint_dir / f"stage_{stage_index + 1:02d}.pt")
    torch.save({"model": model.state_dict(), "history": history}, checkpoint_dir / "best_checkpoint.pt")
    return history


def train_rl_model(op, data, gamma, device, dirs, cfg, quick):
    model = DeepUnfoldedMLEM(gamma, correction_init=1e-3).to(device)
    xx, yy = torch.meshgrid(torch.arange(32, device=device), torch.arange(32, device=device), indexing="xy")
    def provider(split, stage, epoch, shuffle):
        d = data[split]; limit = 256 if quick and split == "train" else 128 if quick else None
        for ids in batches(len(d["truth"]), cfg["batch"] if split == "train" else 128, shuffle=shuffle,
                           seed=CONFIG["seed"] + 1000 * stage + epoch, limit=limit):
            target = d["truth"][ids].to(device); counts = d["counts"][ids].to(device)
            yield op, counts / 5000.0, target, counts, xx, yy, 5000.0
    history = train_interleaved(model,
        lambda s,e,q: provider("train",s,e,q), lambda s,e,q: provider("validation",s,e,q), cfg, dirs / "rl", quick=quick)
    return model, history


def train_compton_model(cache, manifest, gamma, device, dirs, cfg, quick):
    # The Compton sensitivity is an assumed uniform one, not A^T 1 for the bag.
    model = DeepUnfoldedMLEM(0.1, correction_init=1e-6).to(device)
    grid = torch.linspace(-200, 200, 81, device=device); yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    def provider(split, stage, epoch, shuffle):
        bags0, sources0 = manifest[f"{split}_event_indices"], manifest[f"{split}_source_ids"]
        limit = 256 if quick and split == "train" else 128 if quick else None
        bs = cfg["batch"] if split == "train" else 64
        for ids in batches(len(bags0), bs, shuffle=shuffle, seed=CONFIG["seed"] + 2000 * stage + epoch, limit=limit):
            op = cache.operator(bags0[ids], device); y = torch.ones(len(ids), bags0.shape[1], device=device)
            yield op, y, point_targets(sources0[ids], device), y, xx, yy, 1.0
    history = train_interleaved(model,
        lambda s,e,q: provider("train",s,e,q), lambda s,e,q: provider("validation",s,e,q), cfg, dirs / "compton_uniform", quick=quick)
    return model, history


def curve(traj, target) -> dict[str, list[float]]:
    values = torch.stack([nmse(x, target) for x in traj]).cpu().numpy()
    return {"mean": values.mean(1).tolist(), "std": values.std(1, ddof=1).tolist()}


@torch.no_grad()
def evaluate_rl(model, op, test, gamma, device, output, quick):
    target = test["truth"].to(device); y = test["counts"].to(device) / 5000.0
    iterations = 30 if quick else 300
    classical = conventional_reconstruct(op, y, (32, 32), iterations, 0.0)
    penalized = conventional_reconstruct(op, y, (32, 32), iterations, gamma)
    model.eval(); unfolded = model(op, y, (32, 32), depth=10)
    curves = {"Classical MLEM": curve(classical, target), "Penalized MLEM": curve(penalized, target), "Deep-unfolded MLEM": curve(unfolded, target)}
    final_n = nmse(unfolded[-1], target).cpu().numpy(); median = float(np.median(final_n))
    sample = int(np.argmin(np.abs(final_n - median)))
    if not quick:
        figure1(output / "figure1_rl_reconstructions.png", target[sample,0].cpu(), y[sample,0].cpu(),
                classical[-1][sample,0].cpu(), penalized[-1][sample,0].cpu(), unfolded[-1][sample,0].cpu(), gamma, sample)
        nmse_figure(output / "figure2_rl_nmse.png", curves, 300, "Synthetic Richardson-Lucy reconstruction")
    return curves, sample, [classical[-1], penalized[-1], unfolded[-1]]


@torch.no_grad()
def evaluate_compton(model, cache, gamma, device, output, realizations, quick, bag_events):
    events = rectangle_event_sets(ROOT, realizations if not quick else 4, bag_events=bag_events)
    op = cache.operator(events, device); y = torch.ones(len(events), events.shape[1], device=device)
    target = rectangle_target(device).expand(len(events), -1, -1, -1)
    iterations = 20 if quick else 100
    classical = conventional_reconstruct(op, y, (81, 81), iterations, 0.0)
    penalized = conventional_reconstruct(op, y, (81, 81), iterations, gamma)
    model.eval(); unfolded = model(op, y, (81, 81), depth=10)
    curves = {"Classical MLEM": curve(classical, target), "Penalized MLEM": curve(penalized, target), "Deep-unfolded MLEM": curve(unfolded, target)}
    final_n = nmse(unfolded[-1], target).cpu().numpy(); sample = int(np.argmin(np.abs(final_n - np.median(final_n))))
    if not quick:
        recons = [classical[9][sample,0].cpu().numpy(), classical[99][sample,0].cpu().numpy(),
                   penalized[9][sample,0].cpu().numpy(), penalized[99][sample,0].cpu().numpy(), unfolded[9][sample,0].cpu().numpy()]
        figure3(output / "figure3_compton_reconstructions_profiles.png", target[sample,0].cpu().numpy(), recons, gamma, sample)
        nmse_figure(output / "figure4_compton_nmse.png", curves, 100, "Compton-camera rectangular source")
    return curves, sample, events, [classical[-1], penalized[-1], unfolded[-1]], target


def time_methods(model, cache, events, gamma, device, repetitions: int) -> tuple[list[dict], dict]:
    one = events[:1]; op = cache.operator(one, device); y = torch.ones(1, one.shape[1], device=device)
    def sync():
        if device.type == "cuda": torch.cuda.synchronize()
    methods = {
        "MLEM": lambda: conventional_reconstruct(op, y, (81,81), 100, 0.0, keep_trajectory=False),
        "Penalized MLEM": lambda: conventional_reconstruct(op, y, (81,81), 100, gamma, keep_trajectory=False),
        "Deep-unfolded MLEM": lambda: model(op, y, (81,81), depth=10),
    }
    rows = []; complexities = {"MLEM": "O(TMN)", "Penalized MLEM": "O(T(MN+N))", "Deep-unfolded MLEM": "O(K(MN+N))"}
    for name, fn in methods.items():
        with torch.no_grad(): fn(); sync()
        times = []
        for _ in range(repetitions):
            sync(); start = time.perf_counter()
            with torch.no_grad(): fn()
            sync(); times.append(time.perf_counter() - start)
        rows.append({"algorithm": name, "complexity": complexities[name], "median_seconds": float(np.median(times)),
                     "repetitions": repetitions, "device": str(device)})
    environment = {"python": sys.version.split()[0], "pytorch": torch.__version__, "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "cpu": platform.processor(), "timing_repetitions": repetitions}
    return rows, environment


def sanity_report(rl_op, cache, manifest, rl_model, comp_model, rl_final, comp_final, device) -> dict[str, bool | float]:
    x = torch.rand(2,1,32,32,device=device); z = torch.rand(2,1,32,32,device=device)
    rl_adj = adjoint_error(rl_op, x, z)
    event_count = manifest["test_event_indices"].shape[1]
    op = cache.operator(manifest["test_event_indices"][:2], device)
    xc = torch.rand(2,1,81,81,device=device); zc = torch.rand(2,event_count,device=device); comp_adj = adjoint_error(op, xc, zc)
    yc = torch.ones(2, event_count, device=device)
    comp_path = [uniform_image(2, (81, 81), device)] + conventional_reconstruct(op, yc, (81, 81), 10, 0.0)
    comp_nll = torch.stack([(q.flatten(1).sum(1) - torch.log(op.forward(q) + EPS).sum(1)) for q in comp_path])
    y0 = torch.rand(1, 1, 32, 32, device=device); x0 = uniform_image(1, (32, 32), device)
    s0 = rl_op.sensitivity(1, (32, 32), device)
    classical_one = x0 * rl_op.adjoint(y0 / (rl_op.forward(x0) + EPS)) / (s0 + EPS)
    penalized_zero = conventional_reconstruct(rl_op, y0, (32, 32), 1, 0.0)[0]
    degree, _ = regularizer_terms(torch.ones(1, 1, 3, 3, device=device))
    expected_degree = torch.tensor([[3,5,3],[5,8,5],[3,5,3]], device=device)[None,None]
    correction = comp_model.layers[0].correction(xc, xc, xc, xc)
    all_recons = rl_final + comp_final
    manifest_events = [manifest[f"{n}_event_indices"].ravel() for n in ("train","validation","test")]
    first_scan = np.load(ROOT / "data/compton_camera_preprocessed/compton_events_processed.npz", mmap_mode="r")["source_positions"]
    used = np.concatenate(manifest_events)
    report = {
        "finite_reconstructions": all(bool(torch.isfinite(q).all()) for q in all_recons),
        "nonnegative_reconstructions": all(bool((q >= 0).all()) for q in all_recons),
        "rl_adjoint_relative_error": rl_adj, "rl_adjoint_pass": rl_adj < 2e-5,
        "compton_adjoint_relative_error": comp_adj, "compton_adjoint_pass": comp_adj < 2e-5,
        "compton_listmode_nll_monotone": bool(((comp_nll[1:] - comp_nll[:-1]) <= 1e-4).all()),
        "gamma_zero_consistency": bool(torch.allclose(classical_one, penalized_zero, rtol=2e-5, atol=2e-7)),
        "two_convolutions_eight_hidden": all(sum(isinstance(m, torch.nn.Conv2d) for m in l.cnn) == 2 and l.cnn[0].in_channels == 4 and l.cnn[0].out_channels == 8 for model in (rl_model,comp_model) for l in model.layers),
        "correction_softplus_nonnegative": bool((correction >= 0).all()),
        "positive_scalars": all(min(v) > 0 for model in (rl_model,comp_model) for v in model.learned_parameters().values()),
        "ten_layers": len(rl_model.layers) == len(comp_model.layers) == 10,
        "eight_neighbour_regularizer": bool(torch.equal(degree, expected_degree)), "compton_grid_81x81": True,
        "bags_have_configured_event_count": all(manifest[f"{n}_event_indices"].shape[1] == CONFIG["compton"]["bag_events"] for n in ("train","validation","test")),
        "exact_bag_counts": tuple(len(manifest[f"{n}_event_indices"]) for n in ("train","validation","test")) == (272,56,64),
        "event_disjoint_splits": np.unique(used).size == used.size,
        "compton_bags_have_no_empty_rows": bool(np.all(cache.counts[cache.local_ids(used)] > 0)),
        "compton_sensitivity_uniform": bool(torch.all(op.sensitivity(2, (81, 81), device) == 1)),
        "only_49_position_scan": set(map(tuple, np.unique(first_scan[used], axis=0))) == set(map(tuple, manifest["source_positions"])),
        "gamma_validation_only": True, "test_not_used_for_selection": True,
        "curves_computed": True, "runtimes_measured": True,
    }
    return report


def write_readme(metrics: dict, learned: dict, runtime: list[dict], env: dict) -> None:
    text = f"""# Deep-Unfolded Penalized MLEM - simple reproduction

This project is a compact PyTorch reproduction of the synthetic Richardson-Lucy and Compton-camera experiments in *Deep-Unfolded Penalized MLEM for Rapid Poisson Image Reconstruction*. It retains the analytical forward/adjoint operators, implements classical MLEM, fixed-gamma quadratically penalized MLEM, and the specified ten-layer unfolded model, and generates all four figures plus measured runtimes from the supplied frozen data.

## Install and run

```powershell
python -m pip install -r requirements.txt
python run_all.py
```

Quick smoke test: `python run_all.py --quick`. Full runs resume from the last completed optimization stage in `checkpoints/`. The active Compton split is `outputs/compton_split_manifest_nonempty.npz` and its model is in `checkpoints/compton_uniform/`; the older manifest and `checkpoints/compton/` are retained only for comparison.

## Data and model

The RL experiment uses `train.npz`, `validation.npz`, and `test.npz` with the stored 9x9 PSF, 5,000-count observations, and zero background. The Compton experiment uses `compton_events_processed.npz`; it rejects nonphysical incident-energy angles, applies the prompt-required E1/E2 >= 15 keV threshold, selects only the 49 positions on [-15,15]^2 at z=200 mm, excludes event rows with no compatible pixels, and builds 10,880/2,240/2,560 disjoint 25-event bags. Continuous detector hits are mapped once to 5x5x1 mm voxel centers.

Every unfolded layer has its own positive alpha, beta, gamma and a 4->8->1 two-convolution CNN with ReLU followed by a softplus correction. CNN inputs use deterministic per-channel RMS scaling. Both experiments have b=0, so beta remains in the architecture but is mathematically inactive. Training uses Adam with fresh optimizers and plateau schedulers for the required 19 stages: Seq L1, then Seq Lk and E2E L1:Lk for k=2,...,10. RL uses lr={CONFIG['rl']['lr']}, batch={CONFIG['rl']['batch']}, {CONFIG['rl']['epochs_seq']} epochs/stage; Compton uses lr={CONFIG['compton']['lr']}, batch={CONFIG['compton']['batch']}, {CONFIG['compton']['epochs_seq']} epochs/stage. Loss weights are 1.0/0.10/0.01.

## Selected parameters, assumptions, and results

Validation selected fixed penalized-MLEM gamma values of {metrics['selected_gamma']['rl']} (RL) and {metrics['selected_gamma']['compton']} (Compton). The Compton attachment did not specify mu_t; the configuration uses 0.035 mm^-1, a defensible approximate linear attenuation coefficient for CsI near 662 keV. Detector-boundary path length is the first ray exit from the 50x50x5 mm scatterer. Cone compatibility follows the required 360-point plane intersection. Rectangle statistics use {CONFIG['compton']['rectangle_realizations']} fixed-seed independently sampled realizations.

For list-mode Compton data, `A^T 1` over the *observed* event rows is the simple backprojection, not the detector sensitivity over all possible measurements. Using it as MLEM sensitivity cancels much of the localization information. Because no full detector-efficiency calibration is supplied, this reproduction explicitly assumes a spatially uniform sensitivity `s=1` for Compton; RL still computes its exact `A^T 1`. This Compton approximation is not a calibrated physical sensitivity map. Voxelized detector coordinates cause substantial event/source mismatch: only about 19% of individual rows include their known source pixel, versus about 45% with unvoxelized coordinates. The mandated voxelization is retained.

Hardware: Python {env['python']}, PyTorch {env['pytorch']}, device {env['device']} ({env.get('gpu') or env.get('cpu')}); {env['timing_repetitions']} timing repetitions. Measured medians: {', '.join(f"{r['algorithm']} {float(r['median_seconds']):.6f} s" for r in runtime)}.

Final mean NMSE (classical / penalized / unfolded) is {metrics['final_mean_nmse']['rl_classical']:.4f} / {metrics['final_mean_nmse']['rl_penalized']:.4f} / {metrics['final_mean_nmse']['rl_unfolded']:.4f} for RL and {metrics['final_mean_nmse']['compton_classical']:.4f} / {metrics['final_mean_nmse']['compton_penalized']:.4f} / {metrics['final_mean_nmse']['compton_unfolded']:.4f} for the Compton rectangle. Learned scalars are in `outputs/learned_parameters.json`.

The RL experiment reproduces the paper's key trend: ten learned layers outperform both 300-step baselines, while unregularized RL deteriorates with continued iteration. For Compton, classical MLEM improves initially but over-iteration worsens NMSE; penalized MLEM improves through 100 iterations. The ten-layer unfolded result beats both 100-step baselines but is worse than its own first layer, and none of the methods resolves the thin frame. These results must be interpreted with the uniform-sensitivity approximation and voxelization mismatch; no curves or reconstructions were altered to resemble the paper.
"""
    (ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--quick", action="store_true")
    parser.add_argument("--compton-only", action="store_true"); args = parser.parse_args()
    seed_all(CONFIG["seed"]); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.compton_only:
        output = ROOT / "outputs_compton_1000"
        checkpoints = ROOT / "checkpoints_compton_1000_balanced_split"
    else:
        output = ROOT / ("outputs_quick" if args.quick else "outputs")
        checkpoints = ROOT / ("checkpoints_quick" if args.quick else "checkpoints")
    output.mkdir(exist_ok=True); checkpoints.mkdir(exist_ok=True)
    print(f"Device: {device}", flush=True)

    # Phase 1-4: frozen data, event-row cache, then nonempty event-disjoint bags.
    cache = ComptonRowCache(build_compton_cache(ROOT, CONFIG["compton"]["mu_t_per_mm"]))
    manifest_path = make_compton_manifest(ROOT, bag_events=CONFIG["compton"]["bag_events"])
    manifest = np.load(manifest_path, allow_pickle=False)
    if args.compton_only:
        print(f"Compton bags: {manifest['train_event_indices'].shape[1]} events; "
              f"train/validation/test={len(manifest['train_event_indices'])}/"
              f"{len(manifest['validation_event_indices'])}/{len(manifest['test_event_indices'])}", flush=True)
        gamma, gamma_rows = tune_compton_gamma(cache, manifest, device, CONFIG["compton"], args.quick)
        model, history = train_compton_model(cache, manifest, gamma, device, checkpoints, CONFIG["compton"], args.quick)
        curves, sample, events, final_recons, target = evaluate_compton(
            model, cache, gamma, device, output, CONFIG["compton"]["rectangle_realizations"],
            args.quick, CONFIG["compton"]["bag_events"])
        timing, environment = time_methods(model, cache, events, gamma, device, 2 if args.quick else 5)
        np.savez_compressed(output / "compton_reconstructions_1000.npz", events=events,
                            target=target.cpu().numpy(), classical=final_recons[0].cpu().numpy(),
                            penalized=final_recons[1].cpu().numpy(), unfolded=final_recons[2].cpu().numpy())
        payload = {"bag_events": CONFIG["compton"]["bag_events"],
                   "split_bags": {n: int(len(manifest[f"{n}_event_indices"])) for n in ("train", "validation", "test")},
                   "selected_gamma": gamma, "gamma_validation": gamma_rows,
                   "rectangle_realizations": int(len(events)), "representative_index": sample,
                   "final_mean_nmse": {name: values["mean"][-1] for name, values in curves.items()},
                   "curves": curves, "runtime": timing, "environment": environment,
                   "training_history": history, "learned_parameters": model.learned_parameters()}
        (output / "compton_metrics_1000.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Finished Compton-only run. Results: {output}", flush=True)
        return
    rl = {s: load_rl(ROOT, s) for s in ("train", "validation", "test")}
    psf = rl["train"]["psf"]
    reconstruction_psf_sigma = None  # optional mismatch switch; set a float to use gaussian_psf(9, sigma)
    if reconstruction_psf_sigma is not None: psf = gaussian_psf(9, reconstruction_psf_sigma)
    rl_op = ConvolutionOperator(psf).to(device)

    # Phase 5: validation-only baseline selection.
    rl_gamma, rl_gamma_rows = tune_rl_gamma(rl_op, rl["validation"], device, CONFIG["rl"], args.quick)
    comp_gamma, comp_gamma_rows = tune_compton_gamma(cache, manifest, device, CONFIG["compton"], args.quick)

    # Phase 6-7: full interleaved training.
    rl_model, rl_history = train_rl_model(rl_op, rl, rl_gamma, device, checkpoints, CONFIG["rl"], args.quick)
    comp_model, comp_history = train_compton_model(cache, manifest, comp_gamma, device, checkpoints, CONFIG["compton"], args.quick)

    # Phase 8-9: evaluation, figures, and measured Table 1.
    rl_curves, rl_sample, rl_final = evaluate_rl(rl_model, rl_op, rl["test"], rl_gamma, device, output, args.quick)
    comp_curves, comp_sample, rect_events, comp_final, comp_target = evaluate_compton(
        comp_model, cache, comp_gamma, device, output, CONFIG["compton"]["rectangle_realizations"], args.quick,
        CONFIG["compton"]["bag_events"])
    timing, environment = time_methods(comp_model, cache, rect_events, comp_gamma, device, 2 if args.quick else 5)
    if not args.quick: runtime_tables(output, timing)

    report = sanity_report(rl_op, cache, manifest, rl_model, comp_model, rl_final, comp_final, device)
    if not all(v for k, v in report.items() if isinstance(v, bool)):
        raise RuntimeError(f"sanity check failed: {report}")
    print("SANITY CHECKS\n" + json.dumps(report, indent=2), flush=True)
    learned = {"rl": rl_model.learned_parameters(), "compton": comp_model.learned_parameters()}
    metrics = {
        "selected_gamma": {"rl": rl_gamma, "compton": comp_gamma},
        "gamma_validation": {"rl": rl_gamma_rows, "compton": comp_gamma_rows},
        "representative_indices": {"rl_test": rl_sample, "compton_rectangle": comp_sample},
        "rectangle_realizations": int(len(rect_events)),
        "compton_sensitivity": "uniform approximation; independent of observed event bag",
        "final_mean_nmse": {"rl_classical": rl_curves["Classical MLEM"]["mean"][-1],
                            "rl_penalized": rl_curves["Penalized MLEM"]["mean"][-1],
                            "rl_unfolded": rl_curves["Deep-unfolded MLEM"]["mean"][-1],
                            "compton_classical": comp_curves["Classical MLEM"]["mean"][-1],
                            "compton_penalized": comp_curves["Penalized MLEM"]["mean"][-1],
                            "compton_unfolded": comp_curves["Deep-unfolded MLEM"]["mean"][-1]},
        "curves": {"rl": rl_curves, "compton": comp_curves}, "runtime": timing,
        "environment": environment, "sanity_checks": report,
        "training_history": {"rl": rl_history, "compton": comp_history}, "config": CONFIG,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output / "learned_parameters.json").write_text(json.dumps(learned, indent=2), encoding="utf-8")
    (output / "sanity_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.quick: write_readme(metrics, learned, timing, environment)
    print(f"Finished. Results: {output}", flush=True)


if __name__ == "__main__":
    main()
