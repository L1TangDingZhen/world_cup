from __future__ import annotations

import numpy as np

from worldcup_predictor.evaluation.rating_v2_backtest import backtest_rating_v2_poisson
from worldcup_predictor.ingestion.matches import load_matches


def test_rating_v2_backtest_returns_metrics() -> None:
    matches = load_matches("data/examples/synthetic_matches.csv")

    result = backtest_rating_v2_poisson(
        matches,
        cutoff="2025-01-01",
        calibration_bins=5,
    )

    assert result.train_matches > 0
    assert result.test_matches > 0
    assert np.isfinite([result.rps, result.log_loss, result.brier_score]).all()
    assert 0.0 <= result.outcome_accuracy <= 1.0
    assert not result.predictions.empty
    assert not result.calibration.empty
