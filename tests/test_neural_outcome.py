from __future__ import annotations

import numpy as np
import pytest

from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.neural_outcome import (
    NeuralOutcomeConfig,
    NeuralOutcomeModel,
)


def test_neural_outcome_model_predicts_probabilities() -> None:
    matches = load_matches("data/examples/synthetic_matches.csv")
    model = NeuralOutcomeModel(
        config=NeuralOutcomeConfig(epochs=5, hidden_units=8, batch_size=4)
    ).fit(matches, device="cpu")

    probabilities = model.predict_proba("Atlas", "Boreal", device="cpu")

    assert probabilities.shape == (3,)
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities >= 0.0)


def test_neural_outcome_batch_predictions_match_single_predictions() -> None:
    matches = load_matches("data/examples/synthetic_matches.csv")
    model = NeuralOutcomeModel(
        config=NeuralOutcomeConfig(epochs=5, hidden_units=8, batch_size=4)
    ).fit(matches, device="cpu")

    single = model.predict_proba("Atlas", "Boreal", device="cpu")
    batch = model.predict_proba_many(
        [("Atlas", "Boreal", True), ("Comet", "Draco", True)],
        device="cpu",
    )

    assert batch.shape == (2, 3)
    assert np.allclose(batch[0], single)
    assert np.allclose(batch.sum(axis=1), 1.0)


def test_neural_outcome_updates_ratings_after_result() -> None:
    matches = load_matches("data/examples/synthetic_matches.csv")
    model = NeuralOutcomeModel(
        config=NeuralOutcomeConfig(epochs=5, hidden_units=8, batch_size=4)
    ).fit(matches, device="cpu")
    before = dict(model.ratings)

    model.update_ratings("Atlas", "Boreal", 2, 0, "Friendly", False)

    assert model.ratings["Atlas"] != before["Atlas"]
    assert model.ratings["Boreal"] != before["Boreal"]
