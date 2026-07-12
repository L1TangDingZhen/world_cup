from __future__ import annotations

import pandas as pd
import pytest

from worldcup_predictor.ingestion.matches import validate_matches


def test_validate_matches_normalizes_aliases_and_sorts() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2025-02-01", "2025-01-01"],
            "home_team": ["USA", "United States"],
            "away_team": ["Canada", "Mexico"],
            "home_score": [1, 2],
            "away_score": [0, 1],
            "tournament": ["Friendly", "Qualifier"],
            "neutral": ["false", "true"],
        }
    )

    result = validate_matches(raw, team_aliases={"USA": "United States"})

    assert result["date"].is_monotonic_increasing
    assert set(result["home_team"]) == {"United States"}
    assert result["neutral_venue"].tolist() == [True, False]
    assert result["home_goals"].dtype == "int64"


def test_validate_matches_rejects_negative_goals() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2025-01-01"],
            "home_team": ["A"],
            "away_team": ["B"],
            "home_goals": [-1],
            "away_goals": [0],
            "competition_type": ["Friendly"],
            "neutral_venue": [True],
        }
    )

    with pytest.raises(ValueError, match="non-negative integers"):
        validate_matches(raw)


def test_completed_only_drops_unplayed_matches() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-02-01"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_goals": [1, None],
            "away_goals": [0, None],
            "competition_type": ["Friendly", "World Cup"],
            "neutral_venue": [True, True],
        }
    )

    with pytest.raises(ValueError, match="matches without scores"):
        validate_matches(raw)

    result = validate_matches(raw, completed_only=True)

    assert len(result) == 1
    assert result.attrs["dropped_unplayed"] == 1
