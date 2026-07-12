from __future__ import annotations

from worldcup_predictor.evaluation.dixon_coles_backtest import backtest_dixon_coles
from worldcup_predictor.ingestion.matches import load_matches


def test_dixon_coles_backtest_returns_metrics() -> None:
    result = backtest_dixon_coles(
        load_matches("data/examples/synthetic_matches.csv"),
        cutoff="2025-01-01",
        max_iterations=20,
    )

    assert result.train_matches == 10
    assert result.test_matches == 8
    assert result.rps >= 0
