from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.workflows.catch_up import catch_up, fill_fixture_results


def _completed_matches() -> pd.DataFrame:
    return validate_matches(
        pd.DataFrame(
            [
                {
                    "date": "2026-06-11",
                    "home_team": "Alpha",
                    "away_team": "Beta",
                    "home_goals": 2,
                    "away_goals": 0,
                    "competition_type": "FIFA World Cup",
                    "neutral_venue": False,
                },
                {
                    "date": "2026-06-12",
                    "home_team": "Delta",
                    "away_team": "Gamma",
                    "home_goals": 1,
                    "away_goals": 3,
                    "competition_type": "FIFA World Cup",
                    "neutral_venue": True,
                },
            ]
        )
    )


def _write_fixtures(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_fill_fixture_results_fills_exact_and_reversed_orientation(
    tmp_path: Path,
) -> None:
    fixtures_path = tmp_path / "fixtures.csv"
    _write_fixtures(
        fixtures_path,
        [
            {
                "group": "A",
                "date": "2026-06-11",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_goals": None,
                "away_goals": None,
                "neutral_venue": False,
            },
            {
                # Reversed home/away orientation relative to the results data.
                "group": "A",
                "date": "2026-06-12",
                "home_team": "Gamma",
                "away_team": "Delta",
                "home_goals": None,
                "away_goals": None,
                "neutral_venue": True,
            },
            {
                # Not played yet: stays empty.
                "group": "A",
                "date": "2026-06-20",
                "home_team": "Alpha",
                "away_team": "Gamma",
                "home_goals": None,
                "away_goals": None,
                "neutral_venue": True,
            },
        ],
    )

    filled, with_results, total = fill_fixture_results(
        fixtures_path, _completed_matches()
    )

    assert (filled, with_results, total) == (2, 2, 3)
    frame = pd.read_csv(fixtures_path)
    assert frame.loc[0, "home_goals"] == 2
    assert frame.loc[0, "away_goals"] == 0
    assert frame.loc[1, "home_goals"] == 3
    assert frame.loc[1, "away_goals"] == 1
    assert pd.isna(frame.loc[2, "home_goals"])


def test_fill_fixture_results_keeps_existing_scores(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "fixtures.csv"
    _write_fixtures(
        fixtures_path,
        [
            {
                "group": "A",
                "date": "2026-06-11",
                "home_team": "Alpha",
                "away_team": "Beta",
                "home_goals": 5,
                "away_goals": 5,
                "neutral_venue": False,
            }
        ],
    )

    filled, with_results, total = fill_fixture_results(
        fixtures_path, _completed_matches()
    )

    assert (filled, with_results, total) == (0, 1, 1)
    frame = pd.read_csv(fixtures_path)
    assert frame.loc[0, "home_goals"] == 5


def test_catch_up_offline_requires_existing_raw_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        catch_up(
            raw_path=tmp_path / "missing.csv",
            fixtures_path=tmp_path / "fixtures.csv",
            model_output=tmp_path / "model.json",
            offline=True,
        )


def test_catch_up_offline_refits_model_from_local_files(tmp_path: Path) -> None:
    fixtures_path = tmp_path / "fixtures.csv"
    _write_fixtures(
        fixtures_path,
        [
            {
                "group": "A",
                "date": "2030-01-01",
                "home_team": "Atlas",
                "away_team": "Comet",
                "home_goals": None,
                "away_goals": None,
                "neutral_venue": True,
            }
        ],
    )
    model_output = tmp_path / "model.json"

    summary = catch_up(
        raw_path="data/examples/synthetic_matches.csv",
        fixtures_path=fixtures_path,
        model_output=model_output,
        offline=True,
    )

    assert summary.downloaded is False
    assert summary.matches > 0
    assert summary.fixtures_filled_now == 0
    assert summary.trained_through == summary.latest_result_date
    assert model_output.is_file()
