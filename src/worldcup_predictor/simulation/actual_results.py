"""Build knockout-conditioning data from real match results.

Once the group stage is fully recorded in the fixtures file, the bracket is
deterministic, so knockout matches that have already been played in reality
can be pinned to their real winners instead of being re-sampled.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup_predictor.simulation.tournament import TournamentConfig


def load_actual_knockout_winners(
    matches: pd.DataFrame,
    config: TournamentConfig,
    shootouts: pd.DataFrame | None = None,
    competition_type: str = "FIFA World Cup",
) -> dict[frozenset[str], str]:
    """Map each played knockout pairing to its real winner.

    Knockout matches are the tournament matches played after the last group
    fixture between two tournament teams.  A match drawn after full time was
    decided on penalties, so its winner must come from the shootouts data.
    """
    tournament_teams = set(config.groups["team"])
    group_stage_end = config.fixtures["date"].max()
    knockout = matches.loc[
        (matches["competition_type"] == competition_type)
        & (matches["date"] > group_stage_end)
        & matches["home_team"].isin(tournament_teams)
        & matches["away_team"].isin(tournament_teams)
    ]

    shootout_winners: dict[tuple[str, str, str], str] = {}
    if shootouts is not None:
        for row in shootouts.itertuples(index=False):
            key = (
                str(row.date)[:10],
                str(row.home_team).strip(),
                str(row.away_team).strip(),
            )
            shootout_winners[key] = str(row.winner).strip()

    winners: dict[frozenset[str], str] = {}
    for match in knockout.itertuples(index=False):
        home_goals = int(match.home_goals)
        away_goals = int(match.away_goals)
        if home_goals > away_goals:
            winner = match.home_team
        elif away_goals > home_goals:
            winner = match.away_team
        else:
            key = (
                match.date.date().isoformat(),
                match.home_team,
                match.away_team,
            )
            winner = shootout_winners.get(key)
            if winner is None:
                raise ValueError(
                    f"Knockout match {match.home_team} vs {match.away_team} on "
                    f"{key[0]} was drawn but has no shootout winner; download "
                    "or provide the shootouts data."
                )
        winners[frozenset((match.home_team, match.away_team))] = winner
    return winners


def load_knockout_winners_from_files(
    raw_path: str | Path,
    config: TournamentConfig,
    shootouts_path: str | Path | None = None,
    competition_type: str = "FIFA World Cup",
) -> dict[frozenset[str], str]:
    """File-based convenience wrapper for CLI, API and dashboard callers."""
    from worldcup_predictor.ingestion.matches import load_matches

    matches = load_matches(raw_path, completed_only=True)
    shootouts = None
    if shootouts_path is not None and Path(shootouts_path).is_file():
        shootouts = pd.read_csv(shootouts_path)
    return load_actual_knockout_winners(
        matches,
        config,
        shootouts=shootouts,
        competition_type=competition_type,
    )
