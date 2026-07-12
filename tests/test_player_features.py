from __future__ import annotations

import pandas as pd
import pytest

from worldcup_predictor.features.player_features import (
    PlayerAdjustedPredictor,
    aggregate_team_player_features,
    squad_attack_adjustment,
)
from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models import EloPoissonModel


def test_squad_attack_adjustment_handles_missing_data() -> None:
    assert squad_attack_adjustment(pd.DataFrame()) == 0.0
    assert squad_attack_adjustment(pd.DataFrame({"x": [1]})) == 0.0


def test_squad_attack_adjustment_returns_float() -> None:
    value = squad_attack_adjustment(
        pd.DataFrame({"attacking_rating": [70, 80, 95]})
    )

    assert isinstance(value, float)


def test_aggregate_player_features_and_adjust_prediction() -> None:
    players = pd.DataFrame(
        {
            "team": ["Atlas", "Atlas", "Comet"],
            "player": ["A1", "A2", "C1"],
            "attacking_rating": [85, 90, 60],
            "defensive_rating": [80, 82, 65],
            "available": [True, True, True],
        }
    )
    adjustments = aggregate_team_player_features(players)
    base = EloPoissonModel().fit(load_matches("data/examples/synthetic_matches.csv"))
    adjusted = PlayerAdjustedPredictor(base, adjustments).predict("Atlas", "Comet")
    original = base.predict("Atlas", "Comet")

    assert adjustments["Atlas"].available_players == 2
    assert adjusted.expected_home_goals > original.expected_home_goals
    assert (
        adjusted.home_win_prob + adjusted.draw_prob + adjusted.away_win_prob
    ) == pytest.approx(1.0)
