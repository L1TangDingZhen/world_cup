from __future__ import annotations

import time

import numpy as np
import pandas as pd

from worldcup_predictor.compute import ComputeDevice
from worldcup_predictor.evaluation.metrics import (
    brier_score,
    log_loss,
    ranked_probability_score,
)
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.baseline import EloLogisticBaseline, outcome_index
from worldcup_predictor.models.elo_poisson import EloPoissonModel
from worldcup_predictor.models.neural_outcome import (
    NeuralOutcomeConfig,
    NeuralOutcomeModel,
)


def compare_neural_outcome_to_baselines(
    matches: pd.DataFrame,
    cutoff: str | pd.Timestamp,
    neural_config: NeuralOutcomeConfig | None = None,
    neural_device: ComputeDevice = "auto",
) -> pd.DataFrame:
    frame = validate_matches(matches)
    cutoff_timestamp = pd.Timestamp(cutoff)
    train = frame.loc[frame["date"] < cutoff_timestamp].reset_index(drop=True)
    test = frame.loc[frame["date"] >= cutoff_timestamp].reset_index(drop=True)
    if len(train) < 3 or test.empty:
        raise ValueError("Need non-empty chronological train/test splits")

    started = time.perf_counter()
    poisson = EloPoissonModel().fit(train)
    poisson_train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    logistic = EloLogisticBaseline().fit(train)
    logistic_train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    neural = NeuralOutcomeModel(config=neural_config).fit(
        train,
        device=neural_device,
    )
    neural_train_seconds = time.perf_counter() - started

    probabilities_by_model: dict[str, list[np.ndarray]] = {
        "elo_poisson": [],
        "elo_logistic": [],
        "neural_outcome": [],
    }
    outcomes = []
    predict_seconds = {name: 0.0 for name in probabilities_by_model}

    for match in test.itertuples(index=False):
        started = time.perf_counter()
        poisson_prediction = poisson.predict(
            match.home_team,
            match.away_team,
            bool(match.neutral_venue),
        )
        predict_seconds["elo_poisson"] += time.perf_counter() - started
        probabilities_by_model["elo_poisson"].append(
            np.asarray(
                [
                    poisson_prediction.home_win_prob,
                    poisson_prediction.draw_prob,
                    poisson_prediction.away_win_prob,
                ],
                dtype=float,
            )
        )

        started = time.perf_counter()
        probabilities_by_model["elo_logistic"].append(
            logistic.predict_proba(
                match.home_team,
                match.away_team,
                bool(match.neutral_venue),
            )
        )
        predict_seconds["elo_logistic"] += time.perf_counter() - started

        started = time.perf_counter()
        probabilities_by_model["neural_outcome"].append(
            neural.predict_proba(
                match.home_team,
                match.away_team,
                bool(match.neutral_venue),
                device=neural_device,
            )
        )
        predict_seconds["neural_outcome"] += time.perf_counter() - started

        outcome = outcome_index(int(match.home_goals), int(match.away_goals))
        outcomes.append(outcome)
        for model in (poisson, logistic, neural):
            model.update_ratings(
                match.home_team,
                match.away_team,
                int(match.home_goals),
                int(match.away_goals),
                match.competition_type,
                bool(match.neutral_venue),
            )

    outcomes_array = np.asarray(outcomes, dtype=int)
    train_seconds = {
        "elo_poisson": poisson_train_seconds,
        "elo_logistic": logistic_train_seconds,
        "neural_outcome": neural_train_seconds,
    }
    rows = []
    for name, probabilities_list in probabilities_by_model.items():
        probabilities = np.asarray(probabilities_list, dtype=float)
        rows.append(
            {
                "model": name,
                "cutoff": cutoff_timestamp.date().isoformat(),
                "train_matches": len(train),
                "test_matches": len(test),
                "rps": ranked_probability_score(probabilities, outcomes_array),
                "log_loss": log_loss(probabilities, outcomes_array),
                "brier_score": brier_score(probabilities, outcomes_array),
                "outcome_accuracy": float(
                    (probabilities.argmax(axis=1) == outcomes_array).mean()
                ),
                "train_seconds": train_seconds[name],
                "predict_seconds": predict_seconds[name],
                "predictions_per_second": len(test) / predict_seconds[name],
            }
        )
    return pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)
