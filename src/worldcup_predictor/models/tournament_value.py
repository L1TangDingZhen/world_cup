from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.compute import ComputeDevice, resolve_device
from worldcup_predictor.simulation.tournament import TournamentConfig


@dataclass(frozen=True)
class TournamentValueConfig:
    hidden_units: int = 32
    epochs: int = 500
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    seed: int = 42


class TournamentValueNetwork:
    """Experimental value network for tournament champion probabilities.

    It learns a fast approximation of a simulator snapshot:
    team/tournament features -> normalized champion probabilities.
    """

    model_version = "tournament_value_network_v1"

    def __init__(self, config: TournamentValueConfig | None = None) -> None:
        self.config = config or TournamentValueConfig()
        self.network = None
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.group_labels: list[str] = []

    @property
    def is_fitted(self) -> bool:
        return self.network is not None and self.feature_mean is not None

    def fit(
        self,
        ratings: dict[str, float],
        tournament_config: TournamentConfig,
        target_probabilities: pd.DataFrame,
        device: ComputeDevice = "auto",
    ) -> "TournamentValueNetwork":
        torch = _torch()

        teams = tournament_config.groups["team"].tolist()
        target = self._target_vector(target_probabilities, teams)
        if not np.isclose(target.sum(), 1.0, atol=1e-8):
            raise ValueError("Champion probabilities must sum to one")

        self.group_labels = sorted(tournament_config.groups["group"].unique())
        features = self._feature_matrix(ratings, tournament_config)
        self.feature_mean = features.mean(axis=0)
        self.feature_scale = features.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-6] = 1.0
        features = (features - self.feature_mean) / self.feature_scale

        resolved = resolve_device(device)
        torch_device = torch.device(resolved.name)
        torch.manual_seed(self.config.seed)
        if resolved.is_cuda:
            torch.cuda.manual_seed_all(self.config.seed)

        network = self._build_network(input_size=features.shape[1]).to(torch_device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        x = torch.as_tensor(features, dtype=torch.float32, device=torch_device)
        y = torch.as_tensor(target, dtype=torch.float32, device=torch_device)
        network.train()
        for _ in range(self.config.epochs):
            scores = network(x).squeeze(-1)
            log_probabilities = torch.log_softmax(scores, dim=0)
            loss = -(y * log_probabilities).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        network.eval()
        self.network = network.cpu()
        return self

    def predict_champion_probabilities(
        self,
        ratings: dict[str, float],
        tournament_config: TournamentConfig,
        device: ComputeDevice = "auto",
    ) -> pd.DataFrame:
        if self.network is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Model must be fitted before prediction")
        torch = _torch()
        resolved = resolve_device(device)
        torch_device = torch.device(resolved.name)

        features = self._feature_matrix(ratings, tournament_config)
        features = (features - self.feature_mean) / self.feature_scale
        network = self.network.to(torch_device)
        with torch.no_grad():
            scores = network(
                torch.as_tensor(features, dtype=torch.float32, device=torch_device)
            ).squeeze(-1)
            probabilities = torch.softmax(scores, dim=0).cpu().numpy()
        self.network = network.cpu()
        return (
            pd.DataFrame(
                {
                    "team": tournament_config.groups["team"].tolist(),
                    "champion_prob": probabilities.astype(float),
                }
            )
            .sort_values("champion_prob", ascending=False)
            .reset_index(drop=True)
        )

    def to_dict(self) -> dict[str, object]:
        if self.network is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Cannot serialize an unfitted value network")
        return {
            "model_version": self.model_version,
            "config": asdict(self.config),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "group_labels": self.group_labels,
            "state_dict": {
                key: value.detach().cpu().numpy().tolist()
                for key, value in self.network.state_dict().items()
            },
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TournamentValueNetwork":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("model_version") != cls.model_version:
            raise ValueError(f"Unsupported model version: {payload.get('model_version')!r}")

        model = cls(config=TournamentValueConfig(**payload["config"]))
        model.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        model.feature_scale = np.asarray(payload["feature_scale"], dtype=np.float32)
        model.group_labels = [str(value) for value in payload["group_labels"]]
        model.network = model._build_network(input_size=len(model.feature_mean))

        torch = _torch()
        state = {
            key: torch.as_tensor(value, dtype=torch.float32)
            for key, value in payload["state_dict"].items()
        }
        model.network.load_state_dict(state)
        model.network.eval()
        return model

    def _feature_matrix(
        self,
        ratings: dict[str, float],
        tournament_config: TournamentConfig,
    ) -> np.ndarray:
        groups = tournament_config.groups.copy()
        teams = groups["team"].tolist()
        team_ratings = np.asarray([ratings.get(team, 1500.0) for team in teams], dtype=np.float32)
        global_order = np.argsort(np.argsort(-team_ratings)).astype(np.float32)
        global_rank = global_order / max(len(team_ratings) - 1, 1)
        rating_mean = float(team_ratings.mean())
        rating_std = float(team_ratings.std() or 1.0)

        group_stats = {}
        for group, frame in groups.groupby("group", sort=False):
            values = np.asarray([ratings.get(team, 1500.0) for team in frame["team"]], dtype=np.float32)
            ordered = np.argsort(np.argsort(-values)).astype(np.float32)
            group_stats[group] = {
                "mean": float(values.mean()),
                "max": float(values.max()),
                "min": float(values.min()),
                "rank_by_team": {
                    team: float(rank / max(len(values) - 1, 1))
                    for team, rank in zip(frame["team"], ordered, strict=True)
                },
            }

        labels = self.group_labels or sorted(groups["group"].unique())
        features = []
        for index, row in enumerate(groups.itertuples(index=False)):
            rating = float(team_ratings[index])
            stats = group_stats[row.group]
            group_one_hot = [1.0 if row.group == label else 0.0 for label in labels]
            features.append(
                [
                    (rating - 1500.0) / 400.0,
                    (rating - rating_mean) / rating_std,
                    global_rank[index],
                    (stats["mean"] - 1500.0) / 400.0,
                    (stats["max"] - rating) / 400.0,
                    (rating - stats["mean"]) / 400.0,
                    (rating - stats["min"]) / 400.0,
                    stats["rank_by_team"][row.team],
                    *group_one_hot,
                ]
            )
        return np.asarray(features, dtype=np.float32)

    def _build_network(self, input_size: int):
        torch = _torch()
        nn = torch.nn
        return nn.Sequential(
            nn.Linear(input_size, self.config.hidden_units),
            nn.ReLU(),
            nn.Linear(self.config.hidden_units, self.config.hidden_units),
            nn.ReLU(),
            nn.Linear(self.config.hidden_units, 1),
        )

    @staticmethod
    def _target_vector(target_probabilities: pd.DataFrame, teams: list[str]) -> np.ndarray:
        if {"team", "champion_prob"} - set(target_probabilities.columns):
            raise ValueError("target_probabilities must contain team and champion_prob")
        lookup = dict(
            zip(
                target_probabilities["team"].astype(str),
                target_probabilities["champion_prob"].astype(float),
                strict=False,
            )
        )
        target = np.asarray([lookup.get(team, 0.0) for team in teams], dtype=np.float32)
        total = float(target.sum())
        if total <= 0:
            raise ValueError("Champion probabilities must have positive mass")
        return target / total


def fit_value_network_from_target(
    ratings: dict[str, float],
    tournament_config: TournamentConfig,
    target_probabilities: pd.DataFrame,
    value_config: TournamentValueConfig | None = None,
    train_device: ComputeDevice = "auto",
    predict_device: ComputeDevice = "auto",
) -> tuple[TournamentValueNetwork, pd.DataFrame, dict[str, float]]:
    model = TournamentValueNetwork(config=value_config)
    started = time.perf_counter()
    model.fit(
        ratings=ratings,
        tournament_config=tournament_config,
        target_probabilities=target_probabilities,
        device=train_device,
    )
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    prediction = model.predict_champion_probabilities(
        ratings=ratings,
        tournament_config=tournament_config,
        device=predict_device,
    )
    predict_seconds = time.perf_counter() - started
    return model, prediction, {
        "train_seconds": train_seconds,
        "predict_seconds": predict_seconds,
    }


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install.
        raise RuntimeError(
            "TournamentValueNetwork requires PyTorch. Install the project GPU extra or torch."
        ) from exc
    return torch
