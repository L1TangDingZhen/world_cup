from __future__ import annotations

import numpy as np
import pytest

from worldcup_predictor.evaluation.metrics import (
    brier_score,
    calibration_table,
    log_loss,
    ranked_probability_score,
)


def test_perfect_predictions_have_zero_scores() -> None:
    probabilities = np.eye(3)
    outcomes = np.array([0, 1, 2])

    assert ranked_probability_score(probabilities, outcomes) == pytest.approx(0.0)
    assert log_loss(probabilities, outcomes) == pytest.approx(0.0)
    assert brier_score(probabilities, outcomes) == pytest.approx(0.0)


def test_metrics_penalize_wrong_confident_predictions() -> None:
    reasonable = np.array([[0.6, 0.3, 0.1]])
    wrong = np.array([[0.01, 0.09, 0.9]])
    outcome = np.array([0])

    assert log_loss(wrong, outcome) > log_loss(reasonable, outcome)
    assert brier_score(wrong, outcome) > brier_score(reasonable, outcome)
    assert ranked_probability_score(wrong, outcome) > ranked_probability_score(
        reasonable, outcome
    )


def test_calibration_table_accounts_for_all_probabilities() -> None:
    probabilities = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
    outcomes = np.array([0, 2])

    table = calibration_table(probabilities, outcomes, bins=5)

    assert table["count"].sum() == 6

