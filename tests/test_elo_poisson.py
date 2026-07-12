from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.elo_poisson import EloPoissonModel

EXAMPLE_DATA = Path("data/examples/synthetic_matches.csv")


@pytest.fixture
def fitted_model() -> EloPoissonModel:
    return EloPoissonModel().fit(load_matches(EXAMPLE_DATA))


def test_fit_produces_rankings_and_parameters(
    fitted_model: EloPoissonModel,
) -> None:
    assert fitted_model.is_fitted
    assert len(fitted_model.ratings) == 4
    assert fitted_model.parameters is not None
    assert fitted_model.rankings()[0][0] == "Atlas"


def test_prediction_probabilities_sum_to_one(
    fitted_model: EloPoissonModel,
) -> None:
    prediction = fitted_model.predict("Atlas", "Boreal", neutral_venue=True)

    total = (
        prediction.home_win_prob
        + prediction.draw_prob
        + prediction.away_win_prob
    )
    assert total == pytest.approx(1.0)
    assert np.asarray(prediction.score_matrix).sum() == pytest.approx(1.0)
    assert prediction.captured_probability_mass > 0.999
    assert prediction.home_win_prob > prediction.away_win_prob


def test_predict_rejects_unknown_teams_by_default(
    fitted_model: EloPoissonModel,
) -> None:
    with pytest.raises(ValueError, match="No rating for team"):
        fitted_model.predict("Atlas", "Nonexistent Team")

    relaxed = fitted_model.predict(
        "Atlas", "Nonexistent Team", strict_teams=False
    )
    total = relaxed.home_win_prob + relaxed.draw_prob + relaxed.away_win_prob
    assert total == pytest.approx(1.0)


def test_cpu_and_auto_prediction_agree(
    fitted_model: EloPoissonModel,
) -> None:
    cpu_prediction = fitted_model.predict("Atlas", "Boreal", device="cpu")
    auto_prediction = fitted_model.predict("Atlas", "Boreal", device="auto")

    assert auto_prediction.home_win_prob == pytest.approx(
        cpu_prediction.home_win_prob,
        rel=1e-12,
    )
    assert np.allclose(auto_prediction.score_matrix, cpu_prediction.score_matrix)


def test_predict_many_matches_single_prediction_on_cpu(
    fitted_model: EloPoissonModel,
) -> None:
    single = fitted_model.predict("Atlas", "Boreal", device="cpu")
    batch = fitted_model.predict_many(
        [("Atlas", "Boreal", True), ("Comet", "Boreal", True)],
        device="cpu",
    )

    assert batch[0].home_win_prob == pytest.approx(single.home_win_prob)
    assert batch[0].draw_prob == pytest.approx(single.draw_prob)
    assert batch[0].away_win_prob == pytest.approx(single.away_win_prob)
    assert np.allclose(batch[0].score_matrix, single.score_matrix)
    assert batch[1].home_team == "Comet"


def test_predict_many_cuda_matches_cpu_when_available(
    fitted_model: EloPoissonModel,
) -> None:
    from worldcup_predictor.compute import resolve_device

    if resolve_device("auto").name != "cuda":
        pytest.skip("CUDA is not available on this host")

    matches = [("Atlas", "Boreal", True), ("Comet", "Boreal", True)]
    cpu_predictions = fitted_model.predict_many(matches, device="cpu")
    cuda_predictions = fitted_model.predict_many(matches, device="cuda")

    for cpu_prediction, cuda_prediction in zip(
        cpu_predictions,
        cuda_predictions,
        strict=True,
    ):
        assert cuda_prediction.home_win_prob == pytest.approx(
            cpu_prediction.home_win_prob,
            rel=1e-12,
            abs=1e-12,
        )
        assert np.allclose(
            cuda_prediction.score_matrix,
            cpu_prediction.score_matrix,
            rtol=1e-12,
            atol=1e-12,
        )


def test_model_round_trip(tmp_path: Path, fitted_model: EloPoissonModel) -> None:
    destination = tmp_path / "model.json"
    fitted_model.save(destination)
    restored = EloPoissonModel.load(destination)

    original = fitted_model.predict("Atlas", "Comet")
    loaded = restored.predict("Atlas", "Comet")

    assert loaded.home_win_prob == pytest.approx(original.home_win_prob)
    assert loaded.most_likely_score == original.most_likely_score
