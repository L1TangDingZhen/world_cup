from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldcup_predictor.evaluation.metrics import brier_score, log_loss, ranked_probability_score
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.baseline import outcome_index
from worldcup_predictor.models.dixon_coles import DixonColesModel


@dataclass(frozen=True)
class DixonColesBacktestResult:
    cutoff: str
    train_matches: int
    test_matches: int
    rps: float
    log_loss: float
    brier_score: float
    outcome_accuracy: float
    optimization_success: bool | None
    optimization_message: str | None


def backtest_dixon_coles(
    matches: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    max_iterations: int = 5_000,
) -> DixonColesBacktestResult:
    frame = validate_matches(matches)
    cutoff_timestamp = pd.Timestamp(cutoff)
    train = frame.loc[frame["date"] < cutoff_timestamp].reset_index(drop=True)
    test = frame.loc[frame["date"] >= cutoff_timestamp].reset_index(drop=True)
    if len(train) < 3 or test.empty:
        raise ValueError("Need non-empty chronological train/test splits")
    model = DixonColesModel(max_iterations=max_iterations).fit(train)
    probabilities = []
    outcomes = []
    for match in test.itertuples(index=False):
        prediction = model.predict(match.home_team, match.away_team, bool(match.neutral_venue))
        probabilities.append([prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob])
        outcomes.append(outcome_index(int(match.home_goals), int(match.away_goals)))
    p = np.asarray(probabilities)
    y = np.asarray(outcomes)
    return DixonColesBacktestResult(
        cutoff=cutoff_timestamp.date().isoformat(),
        train_matches=len(train),
        test_matches=len(test),
        rps=ranked_probability_score(p, y),
        log_loss=log_loss(p, y),
        brier_score=brier_score(p, y),
        outcome_accuracy=float((p.argmax(axis=1) == y).mean()),
        optimization_success=model.optimization_success,
        optimization_message=model.optimization_message,
    )
