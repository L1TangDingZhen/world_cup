from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldcup_predictor.evaluation.metrics import brier_score, log_loss, ranked_probability_score
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.baseline import EloLogisticBaseline, outcome_index
from worldcup_predictor.models.elo_poisson import EloPoissonModel


@dataclass(frozen=True)
class ModelComparison:
    cutoff: str
    rows: pd.DataFrame


def compare_elo_poisson_to_logistic(
    matches: pd.DataFrame,
    cutoff: str | pd.Timestamp,
) -> ModelComparison:
    frame = validate_matches(matches)
    cutoff_timestamp = pd.Timestamp(cutoff)
    train = frame.loc[frame["date"] < cutoff_timestamp].reset_index(drop=True)
    test = frame.loc[frame["date"] >= cutoff_timestamp].reset_index(drop=True)
    if len(train) < 3 or test.empty:
        raise ValueError("Need non-empty chronological train/test splits")

    poisson = EloPoissonModel().fit(train)
    logistic = EloLogisticBaseline().fit(train)
    poisson_probabilities = []
    logistic_probabilities = []
    outcomes = []
    for match in test.itertuples(index=False):
        prediction = poisson.predict(
            match.home_team,
            match.away_team,
            bool(match.neutral_venue),
            strict_teams=False,
        )
        poisson_probabilities.append(
            [prediction.home_win_prob, prediction.draw_prob, prediction.away_win_prob]
        )
        logistic_probabilities.append(
            logistic.predict_proba(match.home_team, match.away_team, bool(match.neutral_venue))
        )
        outcome = outcome_index(int(match.home_goals), int(match.away_goals))
        outcomes.append(outcome)
        poisson.update_ratings(
            match.home_team, match.away_team, int(match.home_goals), int(match.away_goals),
            match.competition_type, bool(match.neutral_venue),
        )
        logistic.update_ratings(
            match.home_team, match.away_team, int(match.home_goals), int(match.away_goals),
            match.competition_type, bool(match.neutral_venue),
        )

    outcomes_array = np.asarray(outcomes)
    model_probabilities = {
        "elo_poisson": np.asarray(poisson_probabilities),
        "elo_logistic": np.asarray(logistic_probabilities),
    }
    rows = []
    for name, probabilities in model_probabilities.items():
        rows.append(
            {
                "model": name,
                "rps": ranked_probability_score(probabilities, outcomes_array),
                "log_loss": log_loss(probabilities, outcomes_array),
                "brier_score": brier_score(probabilities, outcomes_array),
                "outcome_accuracy": float((probabilities.argmax(axis=1) == outcomes_array).mean()),
                "train_matches": len(train),
                "test_matches": len(test),
            }
        )
    return ModelComparison(cutoff=cutoff_timestamp.date().isoformat(), rows=pd.DataFrame(rows))
