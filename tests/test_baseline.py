from __future__ import annotations

import numpy as np

from worldcup_predictor.evaluation.comparison import (
    compare_elo_poisson_to_dixon_coles,
    compare_elo_poisson_to_logistic,
)
from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.baseline import EloLogisticBaseline


def test_logistic_baseline_predicts_normalized_probabilities() -> None:
    model = EloLogisticBaseline().fit(load_matches("data/examples/synthetic_matches.csv"))
    probabilities = model.predict_proba("Atlas", "Comet")

    assert len(probabilities) == 3
    assert abs(probabilities.sum() - 1.0) < 1e-9


def test_model_comparison_uses_time_split() -> None:
    result = compare_elo_poisson_to_logistic(
        load_matches("data/examples/synthetic_matches.csv"),
        cutoff="2025-01-01",
    )

    assert set(result.rows["model"]) == {"elo_poisson", "elo_logistic"}
    assert set(result.rows["test_matches"]) == {8}


def test_dixon_coles_comparison_refits_on_rolling_schedule() -> None:
    result = compare_elo_poisson_to_dixon_coles(
        load_matches("data/examples/synthetic_matches.csv"),
        cutoff="2025-01-01",
        refit_interval_days=60,
        max_iterations=500,
    )
    rows = result.rows.set_index("model")

    assert set(rows.index) == {"elo_poisson", "dixon_coles"}
    assert set(rows["test_matches"]) == {8}
    # Both models were scored on the same outcomes.
    assert rows.loc["elo_poisson", "actual_draw_rate"] == rows.loc[
        "dixon_coles", "actual_draw_rate"
    ]
    # The 9-month test window at a 60-day cadence must trigger extra refits.
    assert rows.loc["dixon_coles", "refits"] >= 3
    for column in ("rps", "log_loss", "brier_score", "mean_draw_prob"):
        assert np.isfinite(rows[column].astype(float)).all(), column
