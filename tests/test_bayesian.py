from __future__ import annotations

import numpy as np
import pytest

from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.bayesian import BayesianHierarchicalModel


def test_bayesian_hierarchical_model_predicts_distribution() -> None:
    model = BayesianHierarchicalModel(posterior_draws=20, random_seed=1).fit(
        load_matches("data/examples/synthetic_matches.csv")
    )

    prediction = model.predict("Atlas", "Comet")

    assert np.asarray(prediction.score_matrix).sum() == pytest.approx(1.0)
    assert prediction.home_win_prob > 0
    assert prediction.draw_prob > 0
    assert prediction.away_win_prob > 0


def test_bayesian_prediction_interval() -> None:
    model = BayesianHierarchicalModel(posterior_draws=5, random_seed=1).fit(
        load_matches("data/examples/synthetic_matches.csv")
    )

    interval = model.prediction_interval("Atlas", "Comet", draws=20)

    assert 0 <= interval.home_win_low <= interval.home_win_high <= 1
