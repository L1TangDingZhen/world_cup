from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from worldcup_predictor.workflows.dynamic_update import (
    MatchResultInput,
    append_result,
    run_dynamic_update,
    update_fixture_result,
)


def test_append_result_creates_or_extends_csv(tmp_path: Path) -> None:
    path = tmp_path / "manual_results.csv"
    result = MatchResultInput(
        date="2026-06-20",
        home_team="A",
        away_team="B",
        home_goals=1,
        away_goals=0,
        competition_type="Friendly",
        neutral_venue=True,
    )

    append_result(path, result)
    append_result(path, result)

    frame = pd.read_csv(path)
    assert len(frame) == 1
    assert frame.loc[0, "home_team"] == "A"


def test_append_result_preserves_source_column_style(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    pd.DataFrame(
        [
            {
                "date": "2026-06-20",
                "home_team": "A",
                "away_team": "B",
                "home_score": None,
                "away_score": None,
                "tournament": "Friendly",
                "neutral": True,
            }
        ]
    ).to_csv(path, index=False)
    result = MatchResultInput(
        date="2026-06-20",
        home_team="A",
        away_team="B",
        home_goals=1,
        away_goals=0,
        competition_type="Friendly",
        neutral_venue=True,
    )

    assert append_result(path, result) is True
    frame = pd.read_csv(path)
    assert "home_score" in frame.columns
    assert "home_goals" not in frame.columns
    assert frame.loc[0, "home_score"] == 1


def test_update_fixture_result(tmp_path: Path) -> None:
    path = tmp_path / "fixtures.csv"
    pd.DataFrame(
        [
            {
                "group": "A",
                "date": "2026-06-20",
                "home_team": "A",
                "away_team": "B",
                "home_goals": None,
                "away_goals": None,
                "neutral_venue": True,
            }
        ]
    ).to_csv(path, index=False)
    result = MatchResultInput(
        date="2026-06-20",
        home_team="A",
        away_team="B",
        home_goals=2,
        away_goals=1,
        competition_type="Friendly",
        neutral_venue=True,
    )

    assert update_fixture_result(path, result) is True
    frame = pd.read_csv(path)
    assert frame.loc[0, "home_goals"] == 2


def test_run_dynamic_update_refuses_to_refit_from_tiny_dataset(
    tmp_path: Path,
) -> None:
    matches_path = tmp_path / "few_matches.csv"
    pd.DataFrame(
        [
            {
                "date": f"2026-06-{day:02d}",
                "home_team": "A",
                "away_team": "B",
                "home_goals": 1,
                "away_goals": 0,
                "competition_type": "Friendly",
                "neutral_venue": True,
            }
            for day in (1, 2, 3)
        ]
    ).to_csv(matches_path, index=False)
    result = MatchResultInput(
        date="2026-06-20",
        home_team="A",
        away_team="B",
        home_goals=2,
        away_goals=1,
        competition_type="Friendly",
        neutral_venue=True,
    )
    model_output = tmp_path / "model.json"

    with pytest.raises(ValueError, match="Refusing to refit"):
        run_dynamic_update(
            matches_path=matches_path,
            result=result,
            model_output=model_output,
            fixtures_path=tmp_path / "missing_fixtures.csv",
        )

    assert not model_output.exists()
