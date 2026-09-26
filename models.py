"""The three reconstruction equations and the deliberately small unfolded model."""
from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from data_and_operators import EPS, regularizer_terms


def normalize_image(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_min(0) / x.clamp_min(0).sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)


def uniform_image(batch: int, shape: tuple[int, int], device: torch.device) -> torch.Tensor:
    return torch.full((batch, 1, *shape), 1.0 / (shape[0] * shape[1]), device=device)


def nmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    a, b = normalize_image(pred), normalize_image(target)
    return (a - b).square().sum(dim=(-3, -2, -1)) / b.square().sum(dim=(-3, -2, -1)).clamp_min(EPS)


@torch.no_grad()
def conventional_reconstruct(operator, y: torch.Tensor, shape: tuple[int, int], iterations: int, gamma: float, b: torch.Tensor | float = 0.0, *, keep_trajectory: bool = True) -> list[torch.Tensor]:
    """Classical MLEM when gamma=0; quadratically penalized MLEM otherwise."""
    x = uniform_image(y.shape[0], shape, y.device)
    s = operator.sensitivity(y.shape[0], shape, y.device)
    trajectory = []
    for _ in range(iterations):
        p = operator.adjoint(y / (operator.forward(x) + b + EPS))
        u, v = regularizer_terms(x)
        x = (x * (p + gamma * v) / (s + gamma * u + EPS)).clamp_min(0)
        if keep_trajectory or _ == iterations - 1:
            trajectory.append(x.clone())
    return trajectory


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class UnfoldedLayer(nn.Module):
    def __init__(self, gamma_init: float, correction_init: float):
        super().__init__()
        self.raw_alpha = nn.Parameter(torch.tensor(inverse_softplus(1.0)))
        self.raw_beta = nn.Parameter(torch.tensor(inverse_softplus(1.0)))
        self.raw_gamma = nn.Parameter(torch.tensor(inverse_softplus(gamma_init)))
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, stride=1, padding=1),
        )
        nn.init.kaiming_uniform_(self.cnn[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.cnn[0].bias)
        nn.init.normal_(self.cnn[2].weight, std=1e-3)
        nn.init.constant_(self.cnn[2].bias, inverse_softplus(correction_init))

    @property
    def alpha(self) -> torch.Tensor:
        return F.softplus(self.raw_alpha) + 1e-8

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta) + 1e-8

    @property
    def gamma(self) -> torch.Tensor:
        return F.softplus(self.raw_gamma) + 1e-8

    def correction(self, x: torch.Tensor, p: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = torch.cat((x, p, u, v), dim=1)
        # Deterministic per-sample/channel RMS scaling; not a learned module.
        scale = q.square().mean(dim=(-2, -1), keepdim=True).sqrt().detach().clamp_min(EPS)
        return F.softplus(self.cnn(q / scale))


class DeepUnfoldedMLEM(nn.Module):
    def __init__(self, gamma_init: float, correction_init: float, layers: int = 10):
        super().__init__()
        self.layers = nn.ModuleList([UnfoldedLayer(gamma_init, correction_init) for _ in range(layers)])

    def forward(self, operator, y: torch.Tensor, shape: tuple[int, int], b: torch.Tensor | float = 0.0, depth: int = 10) -> list[torch.Tensor]:
        x = uniform_image(y.shape[0], shape, y.device)
        s = operator.sensitivity(y.shape[0], shape, y.device)
        trajectory = []
        for layer in self.layers[:depth]:
            a, be, g = layer.alpha, layer.beta, layer.gamma
            p = operator.adjoint(y / (a * operator.forward(x) + be * b + EPS))
            u, v = regularizer_terms(x)
            c = layer.correction(x, p, u, v)
            x = (x * (a * p + g * v) / (a * s + c + g * u + EPS)).clamp_min(0)
            trajectory.append(x)
        return trajectory

    def learned_parameters(self) -> dict[str, list[float]]:
        return {
            "alpha": [float(x.alpha.detach().cpu()) for x in self.layers],
            "beta": [float(x.beta.detach().cpu()) for x in self.layers],
            "gamma": [float(x.gamma.detach().cpu()) for x in self.layers],
        }


def hybrid_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    operator,
    observed: torch.Tensor,
    layer: UnfoldedLayer,
    coords_x: torch.Tensor,
    coords_y: torch.Tensor,
    *,
    background: torch.Tensor | float = 0.0,
    data_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred_n, target_n = normalize_image(pred), normalize_image(target)
    shape = -(target_n * torch.log(pred_n + EPS)).sum(dim=(-3, -2, -1)).mean()
    px = (pred_n[:, 0] * coords_x).sum(dim=(-2, -1)); py = (pred_n[:, 0] * coords_y).sum(dim=(-2, -1))
    tx = (target_n[:, 0] * coords_x).sum(dim=(-2, -1)); ty = (target_n[:, 0] * coords_y).sum(dim=(-2, -1))
    location = ((px - tx).square() + (py - ty).square()).mean()
    rate = data_scale * (layer.alpha * operator.forward(pred) + layer.beta * background)
    # The expected-count term integrates over all possible measurements.
    # For list-mode Compton data this uses the independent sensitivity, not
    # a sum over only the observed event rows.
    s = operator.sensitivity(pred.shape[0], pred.shape[-2:], pred.device)
    background_total = torch.as_tensor(background, device=pred.device).expand_as(observed).flatten(1).sum(1)
    expected = data_scale * (layer.alpha * (s * pred).flatten(1).sum(1) + layer.beta * background_total)
    data = (expected - (observed * torch.log(rate + EPS)).flatten(1).sum(1)).mean()
    total = shape + 0.10 * location + 0.01 * data
    return total, {"shape": shape, "location": location, "data": data}
