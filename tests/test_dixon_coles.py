from __future__ import annotations

import numpy as np
import pytest

from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.dixon_coles import DixonColesModel, dixon_coles_tau


def test_dixon_coles_tau_only_adjusts_low_scores() -> None:
    assert dixon_coles_tau(2, 1, 1.2, 1.0, -0.05) == 1.0
    assert dixon_coles_tau(0, 0, 1.2, 1.0, -0.05) > 1.0


def test_dixon_coles_prediction_still_normalizes() -> None:
    model = DixonColesModel().fit(load_matches("data/examples/synthetic_matches.csv"))
    prediction = model.predict("Atlas", "Comet")

    assert np.asarray(prediction.score_matrix).sum() == pytest.approx(1.0)
    assert (
        prediction.home_win_prob + prediction.draw_prob + prediction.away_win_prob
    ) == pytest.approx(1.0)
    assert model.attack
    assert model.defense
    assert model.optimization_success is True


def test_dixon_coles_round_trip(tmp_path) -> None:
    model = DixonColesModel().fit(load_matches("data/examples/synthetic_matches.csv"))
    path = tmp_path / "dc.json"
    model.save(path)

    restored = DixonColesModel.load(path)

    assert restored.predict("Atlas", "Comet").home_win_prob == pytest.approx(
        model.predict("Atlas", "Comet").home_win_prob
    )
