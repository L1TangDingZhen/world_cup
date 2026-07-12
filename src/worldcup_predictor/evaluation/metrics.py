from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOME_COUNT = 3


def _validate_inputs(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)

    if probabilities.ndim != 2 or probabilities.shape[1] != OUTCOME_COUNT:
        raise ValueError("probabilities must have shape (n_matches, 3)")
    if outcomes.ndim != 1 or len(outcomes) != len(probabilities):
        raise ValueError("outcomes must have shape (n_matches,)")
    if len(outcomes) == 0:
        raise ValueError("At least one prediction is required")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if (probabilities < 0).any():
        raise ValueError("probabilities cannot be negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("Each probability row must sum to one")
    if not np.isin(outcomes, np.arange(OUTCOME_COUNT)).all():
        raise ValueError("outcomes must contain only 0 (home), 1 (draw), or 2 (away)")
    return probabilities, outcomes


def _one_hot(outcomes: np.ndarray) -> np.ndarray:
    encoded = np.zeros((len(outcomes), OUTCOME_COUNT), dtype=float)
    encoded[np.arange(len(outcomes)), outcomes] = 1.0
    return encoded


def ranked_probability_score(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    probabilities, outcomes = _validate_inputs(probabilities, outcomes)
    observed = _one_hot(outcomes)
    predicted_cumulative = np.cumsum(probabilities, axis=1)[:, :-1]
    observed_cumulative = np.cumsum(observed, axis=1)[:, :-1]
    per_match = np.mean(
        (predicted_cumulative - observed_cumulative) ** 2,
        axis=1,
    )
    return float(per_match.mean())


def log_loss(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    epsilon: float = 1e-15,
) -> float:
    probabilities, outcomes = _validate_inputs(probabilities, outcomes)
    selected = probabilities[np.arange(len(outcomes)), outcomes]
    return float(-np.log(np.clip(selected, epsilon, 1.0)).mean())


def brier_score(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
) -> float:
    probabilities, outcomes = _validate_inputs(probabilities, outcomes)
    observed = _one_hot(outcomes)
    return float(np.mean(np.sum((probabilities - observed) ** 2, axis=1)))


def calibration_table(
    probabilities: np.ndarray,
    outcomes: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    probabilities, outcomes = _validate_inputs(probabilities, outcomes)
    if bins < 2:
        raise ValueError("bins must be at least 2")

    observed = _one_hot(outcomes)
    flattened_probabilities = probabilities.reshape(-1)
    flattened_observed = observed.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(
        np.digitize(flattened_probabilities, edges[1:-1], right=False),
        bins - 1,
    )

    rows: list[dict[str, float | int]] = []
    for bin_index in range(bins):
        selected = assignments == bin_index
        count = int(selected.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_lower": float(edges[bin_index]),
                "bin_upper": float(edges[bin_index + 1]),
                "count": count,
                "mean_predicted_probability": float(
                    flattened_probabilities[selected].mean()
                ),
                "observed_frequency": float(flattened_observed[selected].mean()),
            }
        )
    return pd.DataFrame(rows)

