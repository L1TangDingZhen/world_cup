from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.rating_v2_poisson import RatingV2PoissonModel


EXAMPLE_DATA = Path("data/examples/synthetic_matches.csv")


@pytest.fixture
def fitted_v2_model() -> RatingV2PoissonModel:
    return RatingV2PoissonModel().fit(load_matches(EXAMPLE_DATA))


def test_rating_v2_poisson_fit_predicts_probabilities(
    fitted_v2_model: RatingV2PoissonModel,
) -> None:
    prediction = fitted_v2_model.predict("Atlas", "Boreal", device="cpu")

    assert fitted_v2_model.is_fitted
    assert prediction.home_win_prob + prediction.draw_prob + prediction.away_win_prob == pytest.approx(1.0)
    assert np.asarray(prediction.score_matrix).sum() == pytest.approx(1.0)
    assert prediction.captured_probability_mass > 0.999


def test_rating_v2_poisson_round_trip(
    tmp_path: Path,
    fitted_v2_model: RatingV2PoissonModel,
) -> None:
    destination = tmp_path / "rating_v2.json"
    fitted_v2_model.save(destination)
    restored = RatingV2PoissonModel.load(destination)

    original = fitted_v2_model.predict("Atlas", "Comet", device="cpu")
    loaded = restored.predict("Atlas", "Comet", device="cpu")

    assert loaded.home_win_prob == pytest.approx(original.home_win_prob)
    assert loaded.most_likely_score == original.most_likely_score


def test_rating_v2_poisson_update_changes_state(
    fitted_v2_model: RatingV2PoissonModel,
) -> None:
    before = fitted_v2_model.ratings["Atlas"]

    fitted_v2_model.update_ratings(
        "Atlas",
        "Boreal",
        2,
        0,
        "Friendly",
        False,
    )

    assert fitted_v2_model.ratings["Atlas"] != before
