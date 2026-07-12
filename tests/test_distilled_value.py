from __future__ import annotations

import pytest

from worldcup_predictor.models.tournament_value import TournamentValueConfig
from worldcup_predictor.workflows.distilled_value import (
    predict_distilled_value_engine,
    train_distilled_value_engine,
)


def test_train_distilled_value_engine_saves_and_predicts(tmp_path) -> None:
    value_output = tmp_path / "value.json"
    target_output = tmp_path / "target.csv"
    prediction_output = tmp_path / "prediction.csv"

    _, prediction, summary = train_distilled_value_engine(
        model_path="models/elo_poisson_current.json",
        groups_path="data/worldcup/groups_2026.csv",
        fixtures_path="data/worldcup/fixtures_2026.csv",
        value_model_output=value_output,
        target_output=target_output,
        prediction_output=prediction_output,
        label_simulations=5,
        seed=1,
        value_config=TournamentValueConfig(epochs=5, hidden_units=8),
        label_device="cpu",
        train_device="cpu",
        predict_device="cpu",
    )

    assert value_output.is_file()
    assert target_output.is_file()
    assert prediction_output.is_file()
    assert summary.value_model_output == str(value_output)
    assert prediction["champion_prob"].sum() == pytest.approx(1.0)

    restored = predict_distilled_value_engine(
        model_path="models/elo_poisson_current.json",
        value_model_path=value_output,
        groups_path="data/worldcup/groups_2026.csv",
        fixtures_path="data/worldcup/fixtures_2026.csv",
        device="cpu",
    )

    assert len(restored) == 48
    assert restored["champion_prob"].sum() == pytest.approx(1.0)
