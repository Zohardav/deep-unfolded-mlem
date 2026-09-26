"""Frozen data loading, the two physical operators, and Compton bag construction."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap


EPS = 1e-8
COMPTON_BAG_EVENTS = 1000


def load_rl(root: Path, split: str) -> dict[str, torch.Tensor]:
    z = np.load(root / "data" / "synthetic_thin_rectangle_rl" / f"{split}.npz")
    return {k: torch.from_numpy(z[k].copy()) for k in z.files}


def gaussian_psf(size: int, sigma: float) -> torch.Tensor:
    q = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    yy, xx = torch.meshgrid(q, q, indexing="ij")
    h = torch.exp(-(xx.square() + yy.square()) / (2 * sigma**2))
    return (h / h.sum())[None, None]


class ConvolutionOperator:
    """Zero-padded convolution and its exact transpose."""

    def __init__(self, psf: torch.Tensor):
        self.psf = psf
        self.padding = psf.shape[-1] // 2

    def to(self, device: torch.device) -> "ConvolutionOperator":
        self.psf = self.psf.to(device)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.psf, padding=self.padding)

    def adjoint(self, z: torch.Tensor) -> torch.Tensor:
        return F.conv_transpose2d(z, self.psf, padding=self.padding)

    def sensitivity(self, batch: int, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        return self.adjoint(torch.ones(batch, 1, *shape, device=device))


def regularizer_terms(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """u_n=d_n*x_n and v_n=sum of the true 8-neighbours (including boundaries)."""
    kernel = x.new_ones(1, 1, 3, 3)
    kernel[..., 1, 1] = 0
    v = F.conv2d(x, kernel, padding=1)
    d = F.conv2d(torch.ones_like(x), kernel, padding=1)
    return d * x, v


def nonempty_compton_events(root: Path) -> np.ndarray:
    """Global event IDs with at least one compatible image pixel."""
    cache = root / "cache" / "compton_rows"
    ids = np.load(cache / "event_indices.npy", mmap_mode="r")
    counts = np.load(cache / "counts.npy", mmap_mode="r")
    return np.asarray(ids[counts > 0])


def make_compton_manifest(root: Path, seed: int = 20260920, bag_events: int = COMPTON_BAG_EVENTS) -> Path:
    """Build exact event-disjoint bags from nonempty Compton rows."""
    out = root / "outputs" / f"compton_split_manifest_nonempty_{bag_events}_balanced.npz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    path = root / "data" / "compton_camera_preprocessed" / "compton_events_processed.npz"
    z = np.load(path, mmap_mode="r")
    pos = z["source_positions"]
    e1, e2 = z["first_interaction_energy"], z["second_interaction_energy"]
    valid = z["valid_incident_compton_angle_mask"]
    levels = np.arange(-15.0, 15.1, 5.0)
    intended = (
        (pos[:, 2] == 200.0)
        & np.isin(pos[:, 0], levels)
        & np.isin(pos[:, 1], levels)
        & valid
        & (e1 >= 15.0)
        & (e2 >= 15.0)
    )
    nonempty = np.zeros(len(pos), dtype=bool)
    nonempty[nonempty_compton_events(root)] = True
    intended &= nonempty
    coords = np.array([(x, y, 200.0) for x in levels for y in levels], np.float64)
    rng = np.random.default_rng(seed)
    source_bags = []
    source_global_ids = []
    for sid, xyz in enumerate(coords):
        indices = np.flatnonzero(intended & np.all(pos == xyz, axis=1))
        if len(indices) < 8 * bag_events:
            raise RuntimeError(f"source {xyz.tolist()} has only {len(indices)} eligible events")
        chosen = rng.permutation(indices)[:8 * bag_events].reshape(8, bag_events)
        source_bags.append(chosen)
        source_global_ids.append(np.full(8, sid, np.int16))

    # Keep every source in every split and preserve the original global split ratios.
    train_n = np.full(49, 6)
    val_n = np.ones(49, dtype=np.int64)
    test_n = np.ones(49, dtype=np.int64)
    adjusted = rng.permutation(49)[:22]
    train_n[adjusted] -= 1
    val_n[adjusted[:7]] += 1
    test_n[adjusted[7:]] += 1
    split_bags, split_sources = {k: [] for k in ("train", "validation", "test")}, {k: [] for k in ("train", "validation", "test")}
    for sid, bags in enumerate(source_bags):
        bags = bags[rng.permutation(len(bags))]
        a, b = int(train_n[sid]), int(train_n[sid] + val_n[sid])
        c = int(b + test_n[sid])
        for name, part in (("train", bags[:a]), ("validation", bags[a:b]), ("test", bags[b:c])):
            split_bags[name].append(part)
            split_sources[name].append(np.full(len(part), sid, np.int16))
    payload: dict[str, np.ndarray] = {"source_positions": coords, "seed": np.array(seed)}
    for name in split_bags:
        bags = np.vstack(split_bags[name])
        sources = np.concatenate(split_sources[name])
        order = rng.permutation(len(bags))
        payload[f"{name}_event_indices"] = bags[order]
        payload[f"{name}_source_ids"] = sources[order]
    all_events = np.concatenate([payload[f"{n}_event_indices"].ravel() for n in split_bags])
    expected = (272, 56, 64)
    observed = tuple(len(payload[f"{n}_event_indices"]) for n in split_bags)
    if observed != expected or np.unique(all_events).size != all_events.size:
        raise RuntimeError(f"bad Compton split: counts={observed}, unique={np.unique(all_events).size}/{all_events.size}")
    np.savez_compressed(out, **payload)
    return out


def _voxel_centres(p: np.ndarray, z_low: float) -> np.ndarray:
    """Map continuous hits once to 5 x 5 x 1 mm detector-voxel centres."""
    q = p.copy()
    for axis, (low, step, count) in enumerate(((-25.0, 5.0, 10), (-25.0, 5.0, 10), (z_low, 1.0, 5))):
        idx = np.floor((q[:, axis] - low) / step).astype(np.int64)
        q[:, axis] = low + (np.clip(idx, 0, count - 1) + 0.5) * step
    return q


def _cone_rows(p1: np.ndarray, p2: np.ndarray, theta: np.ndarray, *, mu_t: float, n_phi: int = 360) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 360-ray cone/plane intersections for a chunk of events."""
    b = len(p1)
    axis = p1 - p2  # back toward the source; angle theta is relative to incoming direction
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (b, 1))
    ref[np.abs(axis[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
    e_a = np.cross(axis, ref); e_a /= np.linalg.norm(e_a, axis=1, keepdims=True)
    e_b = np.cross(axis, e_a)
    phi = np.linspace(0.0, 2 * np.pi, n_phi, endpoint=False)
    ring = np.cos(phi)[None, :, None] * e_a[:, None] + np.sin(phi)[None, :, None] * e_b[:, None]
    direction = np.cos(theta)[:, None, None] * axis[:, None] + np.sin(theta)[:, None, None] * ring
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (200.0 - p1[:, None, 2]) / direction[:, :, 2]
    xy = p1[:, None, :2] + t[:, :, None] * direction[:, :, :2]
    ix = np.floor((xy[:, :, 0] + 202.5) / 5.0).astype(np.int32)
    iy = np.floor((xy[:, :, 1] + 202.5) / 5.0).astype(np.int32)
    ok = (t > 0) & np.isfinite(t) & (ix >= 0) & (ix < 81) & (iy >= 0) & (iy < 81)
    ids = np.where(ok, iy * 81 + ix, 6561)
    ids.sort(axis=1)
    unique = (ids < 6561) & np.concatenate([np.ones((b, 1), bool), ids[:, 1:] != ids[:, :-1]], axis=1)
    counts = unique.sum(axis=1).astype(np.uint16)
    cols = np.zeros((b, n_phi), np.int32)
    vals = np.zeros((b, n_phi), np.float32)
    gx = np.linspace(-200.0, 200.0, 81)
    for r in range(b):
        c = ids[r, unique[r]]
        n = len(c)
        if not n:
            continue
        cols[r, :n] = c
        f = np.column_stack((gx[c % 81], gx[c // 81], np.full(n, 200.0)))
        ray = f - p1[r]
        dist = np.linalg.norm(ray, axis=1)
        unit = ray / dist[:, None]
        bounds_lo, bounds_hi = np.array([-25.0, -25.0, -2.5]), np.array([25.0, 25.0, 2.5])
        boundary = np.where(unit >= 0, bounds_hi, bounds_lo)
        with np.errstate(divide="ignore", invalid="ignore"):
            exits = (boundary - p1[r]) / unit
        exits[(exits <= 0) | ~np.isfinite(exits)] = np.inf
        length = exits.min(axis=1)
        vals[r, :n] = (np.exp(-mu_t * length) / dist**2).astype(np.float32)
    return cols, vals, counts


def build_compton_cache(root: Path, mu_t: float = 0.035, chunk: int = 1024) -> Path:
    """Cache compact compatible-pixel rows for every eligible first-scan event."""
    cache = root / "cache" / "compton_rows"
    meta_path = cache / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if abs(meta["mu_t_per_mm"] - mu_t) < 1e-12:
            return cache
    cache.mkdir(parents=True, exist_ok=True)
    z = np.load(root / "data" / "compton_camera_preprocessed" / "compton_events_processed.npz", mmap_mode="r")
    pos = z["source_positions"]; e1 = z["first_interaction_energy"]; e2 = z["second_interaction_energy"]
    levels = np.arange(-15.0, 15.1, 5.0)
    mask = (
        (pos[:, 2] == 200.0) & np.isin(pos[:, 0], levels) & np.isin(pos[:, 1], levels)
        & z["valid_incident_compton_angle_mask"] & (e1 >= 15.0) & (e2 >= 15.0)
    )
    events = np.flatnonzero(mask).astype(np.int64)
    np.save(cache / "event_indices.npy", events)
    cols_mm = open_memmap(cache / "cols.npy", mode="w+", dtype=np.int32, shape=(len(events), 360))
    vals_mm = open_memmap(cache / "weights.npy", mode="w+", dtype=np.float32, shape=(len(events), 360))
    counts_mm = open_memmap(cache / "counts.npy", mode="w+", dtype=np.uint16, shape=(len(events),))
    raw1, raw2 = z["first_interaction_positions"], z["second_interaction_positions"]
    angles = z["incident_compton_angles"]
    for start in range(0, len(events), chunk):
        stop = min(start + chunk, len(events)); idx = events[start:stop]
        p1 = _voxel_centres(np.asarray(raw1[idx]), -2.5)
        p2 = _voxel_centres(np.asarray(raw2[idx]), -52.5)
        c, w, n = _cone_rows(p1, p2, np.asarray(angles[idx]), mu_t=mu_t)
        cols_mm[start:stop], vals_mm[start:stop], counts_mm[start:stop] = c, w, n
        if start % (chunk * 40) == 0:
            print(f"Compton cache: {stop:,}/{len(events):,} events", flush=True)
    cols_mm.flush(); vals_mm.flush(); counts_mm.flush()
    meta = {
        "events": int(len(events)), "max_pixels_per_event": 360, "cone_samples": 360,
        "mu_t_per_mm": mu_t, "voxelization_mm": [5.0, 5.0, 1.0],
        "mean_compatible_pixels": float(np.asarray(counts_mm).mean()),
        "zero_row_count": int(np.count_nonzero(np.asarray(counts_mm) == 0)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return cache


class ComptonRowCache:
    def __init__(self, path: Path):
        self.path = path
        self.events = np.load(path / "event_indices.npy", mmap_mode="r")
        self.cols = np.load(path / "cols.npy", mmap_mode="r")
        self.weights = np.load(path / "weights.npy", mmap_mode="r")
        self.counts = np.load(path / "counts.npy", mmap_mode="r")

    def local_ids(self, global_event_ids: np.ndarray) -> np.ndarray:
        ids = np.searchsorted(self.events, global_event_ids)
        if np.any(ids >= len(self.events)) or not np.array_equal(self.events[ids], global_event_ids):
            raise KeyError("event outside cached 49-source, 15-keV pool")
        return ids

    def operator(self, global_event_ids: np.ndarray, device: torch.device) -> "ComptonBatchOperator":
        local = self.local_ids(np.asarray(global_event_ids))
        cols = torch.from_numpy(np.asarray(self.cols[local]).copy()).long().to(device)
        vals = torch.from_numpy(np.asarray(self.weights[local]).copy()).to(device)
        return ComptonBatchOperator(cols, vals)


class ComptonBatchOperator:
    """Observed event rows with an assumed uniform detection sensitivity."""

    def __init__(self, cols: torch.Tensor, weights: torch.Tensor):
        # Accepted shapes: [B,E,L] or [E,L].
        if cols.ndim == 2:
            cols, weights = cols[None], weights[None]
        self.cols, self.weights = cols, weights
        self.batch, self.events, self.max_hits = cols.shape
        self._sensitivity: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        gathered = torch.gather(flat[:, None, :].expand(-1, self.events, -1), 2, self.cols)
        return (gathered * self.weights).sum(dim=2)

    def adjoint(self, z: torch.Tensor) -> torch.Tensor:
        out = z.new_zeros(self.batch, 6561)
        idx = self.cols.reshape(self.batch, -1)
        contrib = (self.weights * z[:, :, None]).reshape(self.batch, -1)
        out.scatter_add_(1, idx, contrib)
        return out.view(self.batch, 1, 81, 81)

    def sensitivity(self, batch: int, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
        if self._sensitivity is None:
            # List-mode sensitivity integrates all possible detections, not only
            # the observed rows. No full detector calibration is supplied.
            self._sensitivity = torch.ones(self.batch, 1, *shape, device=device)
        return self._sensitivity


def point_targets(source_ids: np.ndarray, device: torch.device) -> torch.Tensor:
    levels = np.arange(-15.0, 15.1, 5.0)
    coords = np.array([(x, y) for x in levels for y in levels])
    grid = np.linspace(-200.0, 200.0, 81)
    yy, xx = np.meshgrid(grid, grid, indexing="ij")
    targets = []
    for x, y in coords[source_ids]:
        q = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 5.0**2))
        targets.append((q / q.sum()).astype(np.float32))
    return torch.from_numpy(np.stack(targets))[:, None].to(device)


def rectangle_target(device: torch.device) -> torch.Tensor:
    """Normalized thin frame made from one single-pixel source at each perimeter location."""
    levels = np.arange(-15.0, 15.1, 5.0)
    image = np.zeros((81, 81), dtype=np.float32)
    for x in levels:
        for y in levels:
            if abs(x) == 15 or abs(y) == 15:
                ix = int(round((x + 200.0) / 5.0))
                iy = int(round((y + 200.0) / 5.0))
                image[iy, ix] = 1.0
    image /= image.sum()
    return torch.from_numpy(image)[None, None].to(device)


def rectangle_event_sets(root: Path, realizations: int, seed: int = 20260922,
                         bag_events: int = COMPTON_BAG_EVENTS) -> np.ndarray:
    z = np.load(root / "data" / "compton_camera_preprocessed" / "compton_events_processed.npz", mmap_mode="r")
    pos, e1, e2 = z["source_positions"], z["first_interaction_energy"], z["second_interaction_energy"]
    levels = np.arange(-15.0, 15.1, 5.0)
    valid = z["valid_incident_compton_angle_mask"] & (e1 >= 15) & (e2 >= 15)
    nonempty = np.zeros(len(pos), dtype=bool)
    nonempty[nonempty_compton_events(root)] = True
    valid &= nonempty
    rng = np.random.default_rng(seed); result = []
    frame_sources = []
    for x in levels:
        for y in levels:
            if abs(x) != 15 and abs(y) != 15:
                continue
            eligible = np.flatnonzero(valid & np.all(pos == np.array([x, y, 200.0]), axis=1))
            frame_sources.append(eligible)
    counts = np.full(len(frame_sources), bag_events // len(frame_sources), dtype=np.int64)
    counts[:bag_events % len(frame_sources)] += 1
    for _ in range(realizations):
        rows = [rng.choice(eligible, int(n), replace=False)
                for eligible, n in zip(frame_sources, counts)]
        result.append(np.concatenate(rows))
    return np.stack(result)
