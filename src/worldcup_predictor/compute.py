"""Runtime device selection and accelerated score-matrix helpers.

The model fit path uses Pandas and SciPy and therefore remains CPU-bound.  This
module keeps the prediction path optional: importing the package does not
require PyTorch unless CUDA is explicitly selected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

ComputeDevice = Literal["auto", "cpu", "cuda"]


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested CUDA backend cannot be used."""


@dataclass(frozen=True)
class ResolvedDevice:
    requested: ComputeDevice
    name: Literal["cpu", "cuda"]

    @property
    def is_cuda(self) -> bool:
        return self.name == "cuda"


def resolve_device(device: ComputeDevice = "auto") -> ResolvedDevice:
    """Resolve a user device choice without importing PyTorch for CPU work."""
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if device == "cpu":
        return ResolvedDevice(requested=device, name="cpu")

    try:
        import torch
    except ImportError as exc:
        if device == "auto":
            return ResolvedDevice(requested=device, name="cpu")
        raise DeviceUnavailableError(
            "CUDA was requested but PyTorch is not installed. "
            "Install a CUDA-enabled PyTorch build in the WC environment."
        ) from exc

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - depends on host driver failures.
        if device == "auto":
            return ResolvedDevice(requested=device, name="cpu")
        raise DeviceUnavailableError(
            "CUDA was requested but PyTorch could not initialize the NVIDIA driver."
        ) from exc

    if available:
        return ResolvedDevice(requested=device, name="cuda")
    if device == "auto":
        return ResolvedDevice(requested=device, name="cpu")
    raise DeviceUnavailableError(
        "CUDA was requested but no usable CUDA device is available. "
        "Check the NVIDIA driver and install a CUDA-enabled PyTorch build."
    )


def score_matrix_from_rates(
    home_rate: float,
    away_rate: float,
    max_goals: int,
    device: ComputeDevice = "auto",
) -> tuple[np.ndarray, float, ResolvedDevice]:
    """Return a normalized independent-Poisson score matrix.

    CUDA uses float64 tensors so its probabilities agree with the CPU backend
    up to normal floating-point rounding.  The returned array is intentionally
    NumPy, because the surrounding simulator and public JSON API consume host
    arrays.
    """
    if home_rate <= 0 or away_rate <= 0:
        raise ValueError("Poisson goal rates must be positive")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")

    resolved = resolve_device(device)
    if resolved.name == "cpu":
        home_probabilities = _poisson_probabilities(home_rate, max_goals)
        away_probabilities = _poisson_probabilities(away_rate, max_goals)
        raw_matrix = np.outer(home_probabilities, away_probabilities)
    else:
        import torch

        goals = torch.arange(
            max_goals + 1,
            device="cuda",
            dtype=torch.float64,
        )
        home_log_probabilities = (
            -home_rate + goals * math.log(home_rate) - torch.lgamma(goals + 1)
        )
        away_log_probabilities = (
            -away_rate + goals * math.log(away_rate) - torch.lgamma(goals + 1)
        )
        raw_matrix = torch.outer(
            torch.exp(home_log_probabilities),
            torch.exp(away_log_probabilities),
        ).cpu().numpy()

    captured_mass = float(raw_matrix.sum())
    if captured_mass <= 0:
        raise RuntimeError("Score matrix has zero probability mass")
    return raw_matrix / captured_mass, captured_mass, resolved


def score_matrices_from_rates(
    home_rates: np.ndarray,
    away_rates: np.ndarray,
    max_goals: int,
    device: ComputeDevice = "auto",
) -> tuple[np.ndarray, np.ndarray, ResolvedDevice]:
    """Return normalized independent-Poisson score matrices for many matches.

    This is intentionally separate from ``score_matrix_from_rates`` so the
    existing single-match CPU path keeps its behavior.  CUDA only becomes useful
    when enough matches are computed in one call to amortize tensor creation and
    host/device transfer costs.
    """
    home_rates = np.asarray(home_rates, dtype=np.float64)
    away_rates = np.asarray(away_rates, dtype=np.float64)
    if home_rates.shape != away_rates.shape:
        raise ValueError("home_rates and away_rates must have the same shape")
    if home_rates.ndim != 1:
        raise ValueError("home_rates and away_rates must be one-dimensional")
    if len(home_rates) == 0:
        raise ValueError("At least one match is required")
    if np.any(home_rates <= 0) or np.any(away_rates <= 0):
        raise ValueError("Poisson goal rates must be positive")
    if max_goals < 0:
        raise ValueError("max_goals must be non-negative")

    resolved = resolve_device(device)
    if resolved.name == "cpu":
        matrices = []
        masses = []
        for home_rate, away_rate in zip(home_rates, away_rates, strict=True):
            matrix, mass, _ = score_matrix_from_rates(
                home_rate=float(home_rate),
                away_rate=float(away_rate),
                max_goals=max_goals,
                device="cpu",
            )
            matrices.append(matrix)
            masses.append(mass)
        return np.stack(matrices), np.asarray(masses, dtype=np.float64), resolved

    import torch

    goals = torch.arange(max_goals + 1, device="cuda", dtype=torch.float64).view(1, -1)
    home = torch.as_tensor(home_rates, device="cuda", dtype=torch.float64).view(-1, 1)
    away = torch.as_tensor(away_rates, device="cuda", dtype=torch.float64).view(-1, 1)
    log_factorial = torch.lgamma(goals + 1)

    home_log_probabilities = -home + goals * torch.log(home) - log_factorial
    away_log_probabilities = -away + goals * torch.log(away) - log_factorial
    raw_matrices = torch.exp(home_log_probabilities).unsqueeze(2) * torch.exp(
        away_log_probabilities
    ).unsqueeze(1)
    masses = raw_matrices.sum(dim=(1, 2))
    normalized = raw_matrices / masses.view(-1, 1, 1)
    return normalized.cpu().numpy(), masses.cpu().numpy(), resolved


def _poisson_probabilities(rate: float, max_goals: int) -> np.ndarray:
    probabilities = np.empty(max_goals + 1, dtype=np.float64)
    probabilities[0] = math.exp(-rate)
    for goals in range(1, max_goals + 1):
        probabilities[goals] = probabilities[goals - 1] * rate / goals
    return probabilities
