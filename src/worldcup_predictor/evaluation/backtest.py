from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldcup_predictor.evaluation.metrics import (
    brier_score,
    calibration_table,
    log_loss,
    ranked_probability_score,
)
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.elo_poisson import EloPoissonModel


@dataclass(frozen=True)
class BacktestResult:
    cutoff: str
    train_matches: int
    test_matches: int
    rps: float
    log_loss: float
    brier_score: float
    predictions: pd.DataFrame
    calibration: pd.DataFrame

    def summary(self) -> dict[str, float | int | str]:
        return {
            "cutoff": self.cutoff,
            "train_matches": self.train_matches,
            "test_matches": self.test_matches,
            "rps": self.rps,
            "log_loss": self.log_loss,
            "brier_score": self.brier_score,
        }


def _outcome_index(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def time_split_backtest(
    matches: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    calibration_bins: int = 10,
) -> BacktestResult:
    frame = validate_matches(matches)
    cutoff_timestamp = pd.Timestamp(cutoff)
    train = frame.loc[frame["date"] < cutoff_timestamp].reset_index(drop=True)
    test = frame.loc[frame["date"] >= cutoff_timestamp].reset_index(drop=True)

    if len(train) < 3:
        raise ValueError("Training split must contain at least three matches")
    if test.empty:
        raise ValueError("Test split must contain at least one match")

    model = EloPoissonModel().fit(train)
    rows: list[dict[str, object]] = []

    for match in test.itertuples(index=False):
        # Teams debuting after the cutoff have no rating yet; let them fall
        # back to the initial rating instead of failing the whole backtest.
        prediction = model.predict(
            home_team=match.home_team,
            away_team=match.away_team,
            neutral_venue=bool(match.neutral_venue),
            strict_teams=False,
        )
        outcome = _outcome_index(int(match.home_goals), int(match.away_goals))
        rows.append(
            {
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_goals": int(match.home_goals),
                "away_goals": int(match.away_goals),
                "home_win_prob": prediction.home_win_prob,
                "draw_prob": prediction.draw_prob,
                "away_win_prob": prediction.away_win_prob,
                "outcome": outcome,
                "most_likely_score": prediction.most_likely_score,
            }
        )

        # Update only after scoring this match, preserving chronological causality.
        model.update_ratings(
            home_team=match.home_team,
            away_team=match.away_team,
            home_goals=int(match.home_goals),
            away_goals=int(match.away_goals),
            competition_type=match.competition_type,
            neutral_venue=bool(match.neutral_venue),
        )

    predictions = pd.DataFrame(rows)
    probabilities = predictions[
        ["home_win_prob", "draw_prob", "away_win_prob"]
    ].to_numpy(dtype=float)
    outcomes = predictions["outcome"].to_numpy(dtype=int)

    return BacktestResult(
        cutoff=cutoff_timestamp.date().isoformat(),
        train_matches=len(train),
        test_matches=len(test),
        rps=ranked_probability_score(probabilities, outcomes),
        log_loss=log_loss(probabilities, outcomes),
        brier_score=brier_score(probabilities, outcomes),
        predictions=predictions,
        calibration=calibration_table(
            probabilities,
            outcomes,
            bins=calibration_bins,
        ),
    )

