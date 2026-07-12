from __future__ import annotations

import numpy as np

from worldcup_predictor.evaluation.neural_outcome import (
    compare_neural_outcome_to_baselines,
)
from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.neural_outcome import NeuralOutcomeConfig


def test_compare_neural_outcome_to_baselines_returns_metrics() -> None:
    matches = load_matches("data/examples/synthetic_matches.csv")

    result = compare_neural_outcome_to_baselines(
        matches,
        cutoff="2025-01-01",
        neural_config=NeuralOutcomeConfig(epochs=3, hidden_units=8, batch_size=4),
        neural_device="cpu",
    )

    assert set(result["model"]) == {
        "elo_poisson",
        "elo_logistic",
        "neural_outcome",
    }
    assert np.isfinite(
        result[["rps", "log_loss", "brier_score", "outcome_accuracy"]].to_numpy()
    ).all()
