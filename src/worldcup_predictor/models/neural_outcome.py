from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from worldcup_predictor.compute import ComputeDevice, resolve_device
from worldcup_predictor.config import EloConfig
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.baseline import outcome_index
from worldcup_predictor.ratings.elo import WorldFootballElo


@dataclass(frozen=True)
class NeuralOutcomeConfig:
    hidden_units: int = 32
    epochs: int = 120
    batch_size: int = 512
    learning_rate: float = 0.01
    weight_decay: float = 1e-4
    seed: int = 42


MatchSpec = tuple[str, str, bool]


class NeuralOutcomeModel:
    """Experimental MLP for home/draw/away probabilities.

    This is a separate research model.  It does not replace the Elo-Poisson
    score model, because it predicts only outcome classes rather than a full
    score distribution.
    """

    model_version = "neural_outcome_v1"

    def __init__(
        self,
        elo_config: EloConfig | None = None,
        config: NeuralOutcomeConfig | None = None,
    ) -> None:
        self.elo_config = elo_config or EloConfig()
        self.config = config or NeuralOutcomeConfig()
        self.ratings: dict[str, float] = {}
        self.feature_mean: np.ndarray | None = None
        self.feature_scale: np.ndarray | None = None
        self.network = None

    @property
    def is_fitted(self) -> bool:
        return self.network is not None and self.feature_mean is not None

    def fit(
        self,
        matches: pd.DataFrame,
        device: ComputeDevice = "auto",
    ) -> "NeuralOutcomeModel":
        torch = _torch()
        nn = torch.nn

        frame = validate_matches(matches)
        if len(frame) < 3:
            raise ValueError("At least three matches are required to fit the model")

        elo = WorldFootballElo(config=self.elo_config)
        features = []
        outcomes = []
        for match in frame.itertuples(index=False):
            features.append(
                self._features_from_ratings(
                    home_team=match.home_team,
                    away_team=match.away_team,
                    neutral_venue=bool(match.neutral_venue),
                    ratings=elo.ratings,
                )
            )
            outcomes.append(outcome_index(int(match.home_goals), int(match.away_goals)))
            elo.update(
                home_team=match.home_team,
                away_team=match.away_team,
                home_goals=int(match.home_goals),
                away_goals=int(match.away_goals),
                competition_type=match.competition_type,
                neutral_venue=bool(match.neutral_venue),
            )

        if len(set(outcomes)) < 2:
            raise ValueError("Neural outcome training requires at least two classes")

        x = np.asarray(features, dtype=np.float32)
        y = np.asarray(outcomes, dtype=np.int64)
        self.feature_mean = x.mean(axis=0)
        self.feature_scale = x.std(axis=0)
        self.feature_scale[self.feature_scale < 1e-6] = 1.0
        x = (x - self.feature_mean) / self.feature_scale

        resolved = resolve_device(device)
        torch_device = torch.device(resolved.name)
        torch.manual_seed(self.config.seed)
        if resolved.is_cuda:
            torch.cuda.manual_seed_all(self.config.seed)

        network = nn.Sequential(
            nn.Linear(x.shape[1], self.config.hidden_units),
            nn.ReLU(),
            nn.Linear(self.config.hidden_units, self.config.hidden_units),
            nn.ReLU(),
            nn.Linear(self.config.hidden_units, 3),
        ).to(torch_device)
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=torch_device)
        y_tensor = torch.as_tensor(y, dtype=torch.long, device=torch_device)
        generator = torch.Generator(device=resolved.name)
        generator.manual_seed(self.config.seed)

        network.train()
        for _ in range(self.config.epochs):
            order = torch.randperm(len(x_tensor), generator=generator, device=torch_device)
            for start in range(0, len(x_tensor), self.config.batch_size):
                batch_index = order[start : start + self.config.batch_size]
                logits = network(x_tensor[batch_index])
                loss = loss_fn(logits, y_tensor[batch_index])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        network.eval()
        self.network = network.cpu()
        self.ratings = dict(elo.ratings)
        return self

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
        device: ComputeDevice = "auto",
    ) -> np.ndarray:
        return self.predict_proba_many(
            [(home_team, away_team, neutral_venue)],
            device=device,
        )[0]

    def predict_proba_many(
        self,
        matches: Sequence[MatchSpec],
        device: ComputeDevice = "auto",
    ) -> np.ndarray:
        if not matches:
            raise ValueError("At least one match is required")
        if self.network is None or self.feature_mean is None or self.feature_scale is None:
            raise RuntimeError("Model must be fitted before prediction")

        torch = _torch()
        resolved = resolve_device(device)
        torch_device = torch.device(resolved.name)
        features = np.asarray(
            [
                self._features_from_ratings(
                    home_team=home_team,
                    away_team=away_team,
                    neutral_venue=neutral_venue,
                    ratings=self.ratings,
                )
                for home_team, away_team, neutral_venue in matches
            ],
            dtype=np.float32,
        )
        features = (features - self.feature_mean) / self.feature_scale
        network = self.network.to(torch_device)
        with torch.no_grad():
            logits = network(torch.as_tensor(features, dtype=torch.float32, device=torch_device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        self.network = network.cpu()
        return probabilities.astype(np.float64)

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        competition_type: str,
        neutral_venue: bool,
    ) -> None:
        elo = WorldFootballElo(config=self.elo_config, ratings=self.ratings)
        elo.update(
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            competition_type=competition_type,
            neutral_venue=neutral_venue,
        )
        self.ratings = dict(elo.ratings)

    def _features_from_ratings(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool,
        ratings: dict[str, float],
    ) -> np.ndarray:
        home_rating = ratings.get(home_team, self.elo_config.initial_rating)
        away_rating = ratings.get(away_team, self.elo_config.initial_rating)
        rating_diff = (home_rating - away_rating) / self.elo_config.rating_scale
        venue = 0.0 if neutral_venue else 1.0
        elo = WorldFootballElo(config=self.elo_config, ratings=ratings)
        expected_home = elo.expected_home_score(home_team, away_team, neutral_venue)
        return np.asarray(
            [
                rating_diff,
                abs(rating_diff),
                venue,
                (home_rating - self.elo_config.initial_rating) / self.elo_config.rating_scale,
                (away_rating - self.elo_config.initial_rating) / self.elo_config.rating_scale,
                expected_home,
            ],
            dtype=np.float32,
        )


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on optional install.
        raise RuntimeError(
            "NeuralOutcomeModel requires PyTorch. Install the project GPU extra or torch."
        ) from exc
    return torch
