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
from worldcup_predictor.models.baseline import outcome_index
from worldcup_predictor.models.rating_v2_poisson import RatingV2PoissonModel


@dataclass(frozen=True)
class RatingV2BacktestResult:
    cutoff: str
    train_matches: int
    test_matches: int
    rps: float
    log_loss: float
    brier_score: float
    outcome_accuracy: float
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
            "outcome_accuracy": self.outcome_accuracy,
        }


def backtest_rating_v2_poisson(
    matches: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    calibration_bins: int = 10,
) -> RatingV2BacktestResult:
    frame = validate_matches(matches)
    cutoff_timestamp = pd.Timestamp(cutoff)
    train = frame.loc[frame["date"] < cutoff_timestamp].reset_index(drop=True)
    test = frame.loc[frame["date"] >= cutoff_timestamp].reset_index(drop=True)
    if len(train) < 3:
        raise ValueError("Training split must contain at least three matches")
    if test.empty:
        raise ValueError("Test split must contain at least one match")

    model = RatingV2PoissonModel().fit(train)
    rows: list[dict[str, object]] = []
    for match in test.itertuples(index=False):
        prediction = model.predict(
            home_team=match.home_team,
            away_team=match.away_team,
            neutral_venue=bool(match.neutral_venue),
            device="cpu",
        )
        outcome = outcome_index(int(match.home_goals), int(match.away_goals))
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
        model.update_ratings(
            home_team=match.home_team,
            away_team=match.away_team,
            home_goals=int(match.home_goals),
            away_goals=int(match.away_goals),
            competition_type=match.competition_type,
            neutral_venue=bool(match.neutral_venue),
            date=match.date,
        )

    predictions = pd.DataFrame(rows)
    probabilities = predictions[
        ["home_win_prob", "draw_prob", "away_win_prob"]
    ].to_numpy(dtype=float)
    outcomes = predictions["outcome"].to_numpy(dtype=int)
    return RatingV2BacktestResult(
        cutoff=cutoff_timestamp.date().isoformat(),
        train_matches=len(train),
        test_matches=len(test),
        rps=ranked_probability_score(probabilities, outcomes),
        log_loss=log_loss(probabilities, outcomes),
        brier_score=brier_score(probabilities, outcomes),
        outcome_accuracy=float((probabilities.argmax(axis=1) == outcomes).mean()),
        predictions=predictions,
        calibration=calibration_table(probabilities, outcomes, bins=calibration_bins),
    )
