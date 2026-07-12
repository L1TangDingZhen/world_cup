from __future__ import annotations

import numpy as np
import pytest

from worldcup_predictor.compute import (
    DeviceUnavailableError,
    resolve_device,
    score_matrix_from_rates,
    score_matrices_from_rates,
)


def test_cpu_score_matrix_is_normalized() -> None:
    matrix, captured_mass, resolved = score_matrix_from_rates(
        home_rate=1.35,
        away_rate=0.95,
        max_goals=8,
        device="cpu",
    )

    assert resolved.name == "cpu"
    assert captured_mass > 0.999
    assert matrix.shape == (9, 9)
    assert matrix.sum() == pytest.approx(1.0)


def test_auto_device_always_resolves_to_a_supported_backend() -> None:
    assert resolve_device("auto").name in {"cpu", "cuda"}


def test_cpu_batch_score_matrices_match_single_matrix_results() -> None:
    home_rates = np.asarray([1.35, 1.8], dtype=np.float64)
    away_rates = np.asarray([0.95, 1.1], dtype=np.float64)

    batch_matrices, batch_masses, resolved = score_matrices_from_rates(
        home_rates,
        away_rates,
        8,
        "cpu",
    )

    assert resolved.name == "cpu"
    assert batch_matrices.shape == (2, 9, 9)
    for index, (home_rate, away_rate) in enumerate(
        zip(home_rates, away_rates, strict=True)
    ):
        single_matrix, single_mass, _ = score_matrix_from_rates(
            float(home_rate),
            float(away_rate),
            8,
            "cpu",
        )
        assert batch_masses[index] == pytest.approx(single_mass)
        assert np.allclose(batch_matrices[index], single_matrix)


def test_explicit_cuda_rejects_unavailable_hardware() -> None:
    if resolve_device("auto").name == "cuda":
        pytest.skip("CUDA is available on this host")

    with pytest.raises(DeviceUnavailableError):
        resolve_device("cuda")


def test_cuda_and_cpu_score_matrices_agree_when_cuda_is_available() -> None:
    if resolve_device("auto").name != "cuda":
        pytest.skip("CUDA is not available on this host")

    cpu_matrix, cpu_mass, _ = score_matrix_from_rates(1.35, 0.95, 8, "cpu")
    cuda_matrix, cuda_mass, resolved = score_matrix_from_rates(1.35, 0.95, 8, "cuda")

    assert resolved.name == "cuda"
    assert cuda_mass == pytest.approx(cpu_mass, rel=1e-12, abs=1e-12)
    assert np.allclose(cuda_matrix, cpu_matrix, rtol=1e-12, atol=1e-12)


def test_cuda_and_cpu_batch_score_matrices_agree_when_cuda_is_available() -> None:
    if resolve_device("auto").name != "cuda":
        pytest.skip("CUDA is not available on this host")

    home_rates = np.asarray([1.35, 1.8, 0.8], dtype=np.float64)
    away_rates = np.asarray([0.95, 1.1, 1.6], dtype=np.float64)
    cpu_matrices, cpu_masses, _ = score_matrices_from_rates(
        home_rates,
        away_rates,
        8,
        "cpu",
    )
    cuda_matrices, cuda_masses, resolved = score_matrices_from_rates(
        home_rates,
        away_rates,
        8,
        "cuda",
    )

    assert resolved.name == "cuda"
    assert np.allclose(cuda_masses, cpu_masses, rtol=1e-12, atol=1e-12)
    assert np.allclose(cuda_matrices, cpu_matrices, rtol=1e-12, atol=1e-12)
