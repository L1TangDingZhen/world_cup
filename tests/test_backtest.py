from __future__ import annotations

from pathlib import Path

import numpy as np

from worldcup_predictor.evaluation.backtest import time_split_backtest
from worldcup_predictor.ingestion.matches import load_matches

EXAMPLE_DATA = Path("data/examples/synthetic_matches.csv")


def test_time_split_backtest_scores_only_future_matches() -> None:
    matches = load_matches(EXAMPLE_DATA)
    result = time_split_backtest(matches, cutoff="2025-01-01", calibration_bins=5)

    assert result.train_matches == 10
    assert result.test_matches == 8
    assert (result.predictions["date"] >= "2025-01-01").all()
    assert np.isfinite([result.rps, result.log_loss, result.brier_score]).all()
    probability_sums = result.predictions[
        ["home_win_prob", "draw_prob", "away_win_prob"]
    ].sum(axis=1)
    assert np.allclose(probability_sums, 1.0)
    assert result.calibration["count"].sum() == result.test_matches * 3

