from __future__ import annotations

import pytest

from worldcup_predictor.models import (
    DixonColesModel,
    EloPoissonModel,
    load_model,
    read_model_version,
)


def test_read_model_version() -> None:
    assert read_model_version("models/elo_poisson_v1.json") == "elo_poisson_v1"
    assert (
        read_model_version("models/dixon_coles_synthetic.json") == "dixon_coles_v1"
    )


def test_load_model_dispatches_on_version() -> None:
    elo = load_model("models/elo_poisson_v1.json")
    dixon_coles = load_model("models/dixon_coles_synthetic.json")

    assert isinstance(elo, EloPoissonModel)
    assert isinstance(dixon_coles, DixonColesModel)
    for model in (elo, dixon_coles):
        prediction = model.predict("Atlas", "Comet", neutral_venue=True)
        total = (
            prediction.home_win_prob
            + prediction.draw_prob
            + prediction.away_win_prob
        )
        assert total == pytest.approx(1.0)


def test_load_model_rejects_unknown_version(tmp_path) -> None:
    path = tmp_path / "model.json"
    path.write_text('{"model_version": "mystery_v9"}', encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported model_version"):
        load_model(path)
