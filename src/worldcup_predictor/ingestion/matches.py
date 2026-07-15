from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

CANONICAL_COLUMNS = (
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "competition_type",
    "neutral_venue",
)

COLUMN_ALIASES = {
    "home_score": "home_goals",
    "away_score": "away_goals",
    "tournament": "competition_type",
    "neutral": "neutral_venue",
}

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n"}


def _parse_boolean(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value for neutral_venue: {value!r}")


def validate_matches(
    matches: pd.DataFrame,
    team_aliases: Mapping[str, str] | None = None,
    completed_only: bool = False,
) -> pd.DataFrame:
    frame = matches.rename(columns=COLUMN_ALIASES).copy()
    missing = sorted(set(CANONICAL_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required match columns: {', '.join(missing)}")

    frame = frame.loc[:, CANONICAL_COLUMNS]
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")

    for column in ("home_team", "away_team", "competition_type"):
        frame[column] = frame[column].astype("string").str.strip()

    if team_aliases:
        frame["home_team"] = frame["home_team"].replace(team_aliases)
        frame["away_team"] = frame["away_team"].replace(team_aliases)

    # Drop unplayed rows before validating team names: upstream data lists
    # scheduled matches whose opponent is still to be determined (empty team,
    # empty scores), and those must not break a completed-only load.
    missing_home_score = frame["home_goals"].isna()
    missing_away_score = frame["away_goals"].isna()
    partially_scored = missing_home_score ^ missing_away_score
    if partially_scored.any():
        rows = frame.index[partially_scored].tolist()
        raise ValueError(f"Rows have only one missing score: {rows}")

    unplayed = missing_home_score & missing_away_score
    dropped_unplayed = int(unplayed.sum())
    if dropped_unplayed:
        if not completed_only:
            raise ValueError(
                f"Found {dropped_unplayed} matches without scores; "
                "use completed_only=True to exclude them"
            )
        frame = frame.loc[~unplayed].copy()

    for column in ("home_team", "away_team", "competition_type"):
        if frame[column].isna().any() or frame[column].eq("").any():
            raise ValueError(f"Column {column} contains empty values")

    for column in ("home_goals", "away_goals"):
        numeric = pd.to_numeric(frame[column], errors="raise")
        if numeric.isna().any() or (numeric < 0).any() or (numeric % 1 != 0).any():
            raise ValueError(f"Column {column} must contain non-negative integers")
        frame[column] = numeric.astype("int64")

    frame["neutral_venue"] = frame["neutral_venue"].map(_parse_boolean).astype(bool)

    same_team = frame["home_team"].eq(frame["away_team"])
    if same_team.any():
        rows = frame.index[same_team].tolist()
        raise ValueError(f"A team cannot play itself; invalid rows: {rows}")

    result = frame.sort_values("date", kind="stable").reset_index(drop=True)
    result.attrs["dropped_unplayed"] = dropped_unplayed
    return result


def load_matches(
    path: str | Path,
    team_aliases: Mapping[str, str] | None = None,
    completed_only: bool = False,
) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Match data file does not exist: {source}")
    return validate_matches(
        pd.read_csv(source),
        team_aliases=team_aliases,
        completed_only=completed_only,
    )
