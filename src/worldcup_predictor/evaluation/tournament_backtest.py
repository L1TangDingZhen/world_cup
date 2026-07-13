"""Full-pipeline tournament backtesting.

The single-match backtests validate match probabilities; this module
validates what the simulator builds on top of them: train a model on data
available before a past tournament, simulate the whole tournament, and
score the per-stage advancement probabilities against what really
happened. The skill scores compare against a "climatology" baseline that
gives every team the structural probability of reaching each stage
(16/32 for the round of 16 of a 32-team World Cup, and so on), so a
positive skill means the model knows more than the bracket structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.dixon_coles import DixonColesModel
from worldcup_predictor.models.elo_poisson import EloPoissonModel
from worldcup_predictor.simulation.formats import TournamentFormat, get_format
from worldcup_predictor.simulation.tournament import (
    TournamentConfig,
    TournamentSimulator,
)

MODEL_TYPES = ("elo_poisson", "dixon_coles")


@dataclass(frozen=True)
class TournamentBacktestResult:
    tournament: str
    model_type: str
    train_before: str
    train_matches: int
    simulations: int
    stage_table: pd.DataFrame
    team_table: pd.DataFrame
    champion_probability: float
    champion_log_score: float
    baseline_champion_log_score: float

    def summary(self) -> dict[str, object]:
        return {
            "tournament": self.tournament,
            "model_type": self.model_type,
            "train_before": self.train_before,
            "train_matches": self.train_matches,
            "simulations": self.simulations,
            "champion_probability": self.champion_probability,
            "champion_log_score": self.champion_log_score,
            "baseline_champion_log_score": self.baseline_champion_log_score,
        }


def load_actual_progress(
    actual_path: str | Path,
    tournament_format: TournamentFormat,
) -> pd.DataFrame:
    """Turn team,furthest rows into 0/1 reached-stage indicators."""
    frame = pd.read_csv(actual_path)
    missing = {"team", "furthest"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing actual-progress columns: {sorted(missing)}")

    stage_columns = tournament_format.stage_columns
    # "furthest" levels beyond the group stage follow the stage columns after
    # the group_qualify alias; qualification and its alias are equivalent.
    levels = ["group", *stage_columns[1:]]
    level_index = {level: position for position, level in enumerate(levels)}

    rows = []
    for record in frame.itertuples(index=False):
        furthest = str(record.furthest).strip()
        if furthest not in level_index:
            raise ValueError(
                f"Unknown furthest stage {furthest!r} for {record.team}; "
                f"expected one of {levels}"
            )
        reached = level_index[furthest]
        indicators = {
            stage: int(level_index[stage] <= reached)
            for stage in stage_columns[1:]
        }
        indicators["group_qualify"] = indicators[
            tournament_format.qualification_stage
        ]
        rows.append({"team": str(record.team).strip(), **indicators})
    return pd.DataFrame(rows)


def stage_probability_table(
    simulation: pd.DataFrame,
    actual: pd.DataFrame,
    tournament_format: TournamentFormat,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score simulated stage probabilities against the real progression."""
    predicted = simulation.set_index("team")
    observed = actual.set_index("team")
    missing_teams = set(predicted.index) ^ set(observed.index)
    if missing_teams:
        raise ValueError(
            f"Simulation and actual progress disagree on teams: {sorted(missing_teams)}"
        )
    observed = observed.reindex(predicted.index)

    team_count = len(predicted)
    structural = tournament_format.teams_reaching
    stage_rows = []
    team_table = pd.DataFrame({"team": predicted.index})
    for stage in tournament_format.stage_columns:
        probabilities = predicted[f"{stage}_prob"].to_numpy(dtype=float)
        outcomes = observed[stage].to_numpy(dtype=float)
        baseline_probability = structural[stage] / team_count
        brier = float(np.mean((probabilities - outcomes) ** 2))
        baseline_brier = float(
            np.mean((baseline_probability - outcomes) ** 2)
        )
        stage_rows.append(
            {
                "stage": stage,
                "teams": team_count,
                "actual_count": int(outcomes.sum()),
                "predicted_sum": float(probabilities.sum()),
                "brier": brier,
                "baseline_brier": baseline_brier,
                "skill": 1.0 - brier / baseline_brier if baseline_brier else np.nan,
            }
        )
        team_table[f"{stage}_prob"] = probabilities
        team_table[f"{stage}_actual"] = outcomes.astype(int)
    return pd.DataFrame(stage_rows), team_table


def backtest_tournament(
    matches: pd.DataFrame,
    groups_path: str | Path,
    fixtures_path: str | Path,
    actual_path: str | Path,
    format_name: str,
    train_before: str | pd.Timestamp,
    tournament_label: str | None = None,
    model_type: str = "elo_poisson",
    simulations: int = 10_000,
    seed: int = 42,
    dixon_coles_window_days: int = 3650,
    dixon_coles_max_iterations: int = 3_000,
) -> TournamentBacktestResult:
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")
    frame = validate_matches(matches)
    cutoff = pd.Timestamp(train_before)
    train = frame.loc[frame["date"] < cutoff].reset_index(drop=True)
    if len(train) < 3:
        raise ValueError("Training split must contain at least three matches")

    if model_type == "elo_poisson":
        model = EloPoissonModel().fit(train)
    else:
        window_start = cutoff - pd.Timedelta(days=dixon_coles_window_days)
        model = DixonColesModel(max_iterations=dixon_coles_max_iterations).fit(
            train.loc[train["date"] >= window_start]
        )

    tournament_format = get_format(format_name)
    config = TournamentConfig.from_csv(
        groups_path,
        fixtures_path,
        format_name=format_name,
    )
    simulation = TournamentSimulator(
        predictor=model,
        config=config,
        random_seed=seed,
    ).run(simulations=simulations)

    actual = load_actual_progress(actual_path, tournament_format)
    stage_table, team_table = stage_probability_table(
        simulation, actual, tournament_format
    )

    champion_row = team_table.loc[team_table["champion_actual"] == 1]
    if len(champion_row) != 1:
        raise ValueError("Actual progress must contain exactly one champion")
    champion_probability = float(champion_row["champion_prob"].iloc[0])
    minimum_probability = 1.0 / (simulations * 10.0)
    team_count = len(team_table)
    return TournamentBacktestResult(
        tournament=tournament_label or str(Path(str(groups_path)).stem),
        model_type=model_type,
        train_before=cutoff.date().isoformat(),
        train_matches=len(train),
        simulations=simulations,
        stage_table=stage_table,
        team_table=team_table.sort_values(
            "champion_prob", ascending=False
        ).reset_index(drop=True),
        champion_probability=champion_probability,
        champion_log_score=-math.log(max(champion_probability, minimum_probability)),
        baseline_champion_log_score=-math.log(1.0 / team_count),
    )
