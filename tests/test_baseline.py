from __future__ import annotations

from worldcup_predictor.evaluation.comparison import compare_elo_poisson_to_logistic
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
