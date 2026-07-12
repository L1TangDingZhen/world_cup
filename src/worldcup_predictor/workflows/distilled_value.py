from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.compute import ComputeDevice, resolve_device
from worldcup_predictor.models.elo_poisson import EloPoissonModel
from worldcup_predictor.models.tournament_value import (
    TournamentValueConfig,
    TournamentValueNetwork,
)
from worldcup_predictor.simulation.batch_tournament import (
    load_batch_elo_poisson_simulator,
)
from worldcup_predictor.simulation.tournament import TournamentConfig


@dataclass(frozen=True)
class DistilledValueSummary:
    label_simulations: int
    label_device: str
    train_device: str
    predict_device: str
    label_seconds: float
    train_seconds: float
    predict_seconds: float
    mae: float
    rmse: float
    max_abs_error: float
    target_top_champion: str
    target_top_champion_prob: float
    value_top_champion: str
    value_top_champion_prob: float
    value_model_output: str | None = None
    target_output: str | None = None
    prediction_output: str | None = None

    def to_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def train_distilled_value_engine(
    model_path: str | Path,
    groups_path: str | Path,
    fixtures_path: str | Path,
    value_model_output: str | Path | None = None,
    target_output: str | Path | None = None,
    prediction_output: str | Path | None = None,
    label_simulations: int = 2000,
    seed: int | None = None,
    value_config: TournamentValueConfig | None = None,
    label_device: ComputeDevice = "cpu",
    train_device: ComputeDevice = "cpu",
    predict_device: ComputeDevice = "cpu",
) -> tuple[TournamentValueNetwork, pd.DataFrame, DistilledValueSummary]:
    if label_simulations <= 0:
        raise ValueError("label_simulations must be positive")

    base_model = EloPoissonModel.load(model_path)
    tournament_config = TournamentConfig.from_csv(groups_path, fixtures_path)
    simulator = load_batch_elo_poisson_simulator(
        model_path=model_path,
        groups_path=groups_path,
        fixtures_path=fixtures_path,
        random_seed=seed,
        device=label_device,
    )

    started = time.perf_counter()
    target = simulator.run(simulations=label_simulations)
    label_seconds = time.perf_counter() - started

    value_model = TournamentValueNetwork(config=value_config)
    started = time.perf_counter()
    value_model.fit(
        ratings=base_model.ratings,
        tournament_config=tournament_config,
        target_probabilities=target[["team", "champion_prob"]],
        device=train_device,
    )
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prediction = value_model.predict_champion_probabilities(
        ratings=base_model.ratings,
        tournament_config=tournament_config,
        device=predict_device,
    )
    predict_seconds = time.perf_counter() - started

    merged = prediction.merge(
        target[["team", "champion_prob"]],
        on="team",
        suffixes=("_value", "_target"),
    )
    error = (
        merged["champion_prob_value"] - merged["champion_prob_target"]
    ).to_numpy(dtype=float)

    saved_value_model = _save_value_model(value_model, value_model_output)
    saved_target = _save_frame(target, target_output)
    saved_prediction = _save_frame(prediction, prediction_output)

    summary = DistilledValueSummary(
        label_simulations=label_simulations,
        label_device=resolve_device(label_device).name,
        train_device=resolve_device(train_device).name,
        predict_device=resolve_device(predict_device).name,
        label_seconds=label_seconds,
        train_seconds=train_seconds,
        predict_seconds=predict_seconds,
        mae=float(np.mean(np.abs(error))),
        rmse=float(np.sqrt(np.mean(error**2))),
        max_abs_error=float(np.max(np.abs(error))),
        target_top_champion=str(target.iloc[0]["team"]),
        target_top_champion_prob=float(target.iloc[0]["champion_prob"]),
        value_top_champion=str(prediction.iloc[0]["team"]),
        value_top_champion_prob=float(prediction.iloc[0]["champion_prob"]),
        value_model_output=saved_value_model,
        target_output=saved_target,
        prediction_output=saved_prediction,
    )
    return value_model, prediction, summary


def predict_distilled_value_engine(
    model_path: str | Path,
    value_model_path: str | Path,
    groups_path: str | Path,
    fixtures_path: str | Path,
    device: ComputeDevice = "cpu",
) -> pd.DataFrame:
    base_model = EloPoissonModel.load(model_path)
    value_model = TournamentValueNetwork.load(value_model_path)
    tournament_config = TournamentConfig.from_csv(groups_path, fixtures_path)
    return value_model.predict_champion_probabilities(
        ratings=base_model.ratings,
        tournament_config=tournament_config,
        device=device,
    )


def _save_value_model(
    value_model: TournamentValueNetwork,
    path: str | Path | None,
) -> str | None:
    if path is None:
        return None
    value_model.save(path)
    return str(path)


def _save_frame(frame: pd.DataFrame, path: str | Path | None) -> str | None:
    if path is None:
        return None
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return str(destination)
