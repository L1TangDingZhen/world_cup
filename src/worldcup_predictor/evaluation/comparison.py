from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from worldcup_predictor.evaluation.metrics import brier_score, log_loss, ranked_probability_score
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.baseline import EloLogisticBaseline, outcome_index
from worldcup_predictor.models.dixon_coles import DixonColesModel
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


def compare_elo_poisson_to_dixon_coles(
    matches: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    refit_interval_days: int = 30,
    training_window_days: int = 3650,
    max_iterations: int = 2_000,
) -> ModelComparison:
    """Compare Elo-Poisson and Dixon-Coles under equally dynamic protocols.

    A static Dixon-Coles fit frozen at the cutoff would go stale over a long
    test window while the Elo ratings keep updating, so this comparison
    refits Dixon-Coles on a rolling window every ``refit_interval_days``:
    both models only ever see matches played before the one being predicted.
    """
    frame = validate_matches(matches)
    cutoff_timestamp = pd.Timestamp(cutoff)
    train = frame.loc[frame["date"] < cutoff_timestamp].reset_index(drop=True)
    test = frame.loc[frame["date"] >= cutoff_timestamp].reset_index(drop=True)
    if len(train) < 3 or test.empty:
        raise ValueError("Need non-empty chronological train/test splits")

    elo_poisson = EloPoissonModel().fit(train)

    window = pd.Timedelta(days=training_window_days)
    refit_interval = pd.Timedelta(days=refit_interval_days)

    def fit_dixon_coles(refit_date: pd.Timestamp) -> DixonColesModel:
        training = frame.loc[
            (frame["date"] >= refit_date - window) & (frame["date"] < refit_date)
        ]
        return DixonColesModel(max_iterations=max_iterations).fit(training)

    dixon_coles = fit_dixon_coles(cutoff_timestamp)
    next_refit = cutoff_timestamp + refit_interval
    refits = 1

    poisson_probabilities: list[list[float]] = []
    dixon_coles_probabilities: list[list[float]] = []
    outcomes: list[int] = []
    for match in test.itertuples(index=False):
        if match.date >= next_refit:
            dixon_coles = fit_dixon_coles(match.date)
            next_refit = match.date + refit_interval
            refits += 1

        poisson_prediction = elo_poisson.predict(
            match.home_team,
            match.away_team,
            bool(match.neutral_venue),
            strict_teams=False,
        )
        poisson_probabilities.append(
            [
                poisson_prediction.home_win_prob,
                poisson_prediction.draw_prob,
                poisson_prediction.away_win_prob,
            ]
        )
        dixon_coles_prediction = dixon_coles.predict(
            match.home_team,
            match.away_team,
            bool(match.neutral_venue),
        )
        dixon_coles_probabilities.append(
            [
                dixon_coles_prediction.home_win_prob,
                dixon_coles_prediction.draw_prob,
                dixon_coles_prediction.away_win_prob,
            ]
        )
        outcomes.append(outcome_index(int(match.home_goals), int(match.away_goals)))
        elo_poisson.update_ratings(
            match.home_team,
            match.away_team,
            int(match.home_goals),
            int(match.away_goals),
            match.competition_type,
            bool(match.neutral_venue),
        )

    outcomes_array = np.asarray(outcomes)
    actual_draw_rate = float((outcomes_array == 1).mean())
    rows = []
    for name, probabilities, model_refits in (
        ("elo_poisson", np.asarray(poisson_probabilities), None),
        ("dixon_coles", np.asarray(dixon_coles_probabilities), refits),
    ):
        rows.append(
            {
                "model": name,
                "rps": ranked_probability_score(probabilities, outcomes_array),
                "log_loss": log_loss(probabilities, outcomes_array),
                "brier_score": brier_score(probabilities, outcomes_array),
                "outcome_accuracy": float(
                    (probabilities.argmax(axis=1) == outcomes_array).mean()
                ),
                "mean_draw_prob": float(probabilities[:, 1].mean()),
                "actual_draw_rate": actual_draw_rate,
                "refits": model_refits,
                "train_matches": len(train),
                "test_matches": len(test),
            }
        )
    return ModelComparison(cutoff=cutoff_timestamp.date().isoformat(), rows=pd.DataFrame(rows))
