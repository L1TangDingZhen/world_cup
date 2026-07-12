from __future__ import annotations

import pandas as pd
import pytest

from worldcup_predictor.ratings.v2 import FootballRatingV2, RatingV2Config


def test_rating_v2_update_tracks_uncertainty_and_recent_form() -> None:
    engine = FootballRatingV2()

    home_after, away_after = engine.update(
        home_team="A",
        away_team="B",
        home_goals=3,
        away_goals=0,
        competition_type="World Cup",
        neutral_venue=True,
        date=pd.Timestamp("2024-01-01"),
    )

    assert home_after > 1500
    assert away_after < 1500
    assert engine.states["A"].uncertainty < RatingV2Config().initial_uncertainty
    assert engine.states["A"].recent_form > 0
    assert engine.states["B"].recent_form < 0


def test_rating_v2_uncertainty_grows_when_team_is_inactive() -> None:
    engine = FootballRatingV2()
    engine.update(
        "A",
        "B",
        1,
        0,
        "Friendly",
        True,
        date=pd.Timestamp("2024-01-01"),
    )
    after_match = engine.states["A"].uncertainty

    aged = engine.state_for("A", pd.Timestamp("2026-01-01"))

    assert aged.uncertainty > after_match
    assert aged.uncertainty <= RatingV2Config().max_uncertainty


def test_rating_v2_goal_difference_multiplier_is_log_scaled() -> None:
    assert FootballRatingV2.goal_difference_multiplier(1) == pytest.approx(1.0)
    assert FootballRatingV2.goal_difference_multiplier(3) > 1.0
    assert FootballRatingV2.goal_difference_multiplier(6) < 4.0
