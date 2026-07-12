from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from worldcup_predictor.compute import (
    ComputeDevice,
    score_matrix_from_rates,
    score_matrices_from_rates,
)
from worldcup_predictor.config import EloConfig, ModelConfig
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.ratings.elo import WorldFootballElo


@dataclass(frozen=True)
class PoissonParameters:
    base_log_goal_rate: float
    elo_coefficient: float
    home_advantage: float
    time_decay_xi: float


@dataclass(frozen=True)
class MatchPrediction:
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    most_likely_score: str
    score_matrix: list[list[float]]
    captured_probability_mass: float

    def to_dict(self, include_score_matrix: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_score_matrix:
            result.pop("score_matrix")
        return result


type MatchSpec = tuple[str, str, bool]


class EloPoissonModel:
    model_version = "elo_poisson_v1"

    def __init__(
        self,
        elo_config: EloConfig | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.elo_config = elo_config or EloConfig()
        self.model_config = model_config or ModelConfig()
        self.ratings: dict[str, float] = {}
        self.parameters: PoissonParameters | None = None
        self.trained_through: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.parameters is not None and bool(self.ratings)

    def fit(self, matches: pd.DataFrame) -> "EloPoissonModel":
        frame = validate_matches(matches)
        if len(frame) < 3:
            raise ValueError("At least three matches are required to fit the model")

        elo = WorldFootballElo(config=self.elo_config)
        history = elo.process(frame)

        rating_difference = (
            history["home_elo_before"].to_numpy(dtype=float)
            - history["away_elo_before"].to_numpy(dtype=float)
        ) / self.elo_config.rating_scale
        home_indicator = (~history["neutral_venue"]).to_numpy(dtype=float)
        home_goals = history["home_goals"].to_numpy(dtype=float)
        away_goals = history["away_goals"].to_numpy(dtype=float)

        latest_date = history["date"].max()
        age_days = (latest_date - history["date"]).dt.days.to_numpy(dtype=float)
        xi = math.log(2.0) / self.model_config.time_decay_half_life_days
        weights = np.exp(-xi * age_days)

        mean_goals = max((home_goals.sum() + away_goals.sum()) / (2 * len(frame)), 0.1)
        initial = np.array([math.log(mean_goals), 0.35, 0.10], dtype=float)

        def objective(values: np.ndarray) -> float:
            base, coefficient, home_advantage = values
            home_log_rate = np.clip(
                base
                + coefficient * rating_difference
                + home_advantage * home_indicator,
                -5.0,
                5.0,
            )
            away_log_rate = np.clip(
                base - coefficient * rating_difference,
                -5.0,
                5.0,
            )
            home_rate = np.exp(home_log_rate)
            away_rate = np.exp(away_log_rate)
            negative_log_likelihood = weights * (
                home_rate
                - home_goals * home_log_rate
                + away_rate
                - away_goals * away_log_rate
            )
            return float(negative_log_likelihood.sum() / weights.sum())

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=((-3.0, 2.0), (0.0, 3.0), (-1.0, 1.0)),
        )
        if not result.success:
            raise RuntimeError(f"Poisson parameter fitting failed: {result.message}")

        self.parameters = PoissonParameters(
            base_log_goal_rate=float(result.x[0]),
            elo_coefficient=float(result.x[1]),
            home_advantage=float(result.x[2]),
            time_decay_xi=xi,
        )
        self.ratings = dict(elo.ratings)
        self.trained_through = latest_date.date().isoformat()
        return self

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
        strict_teams: bool = True,
    ) -> tuple[float, float]:
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before prediction")

        if strict_teams:
            unknown = [
                team for team in (home_team, away_team) if team not in self.ratings
            ]
            if unknown:
                raise ValueError(
                    f"No rating for team(s): {', '.join(unknown)}. Check the "
                    "spelling against the training data, or pass "
                    "strict_teams=False to use the initial rating."
                )
        home_rating = self.ratings.get(home_team, self.elo_config.initial_rating)
        away_rating = self.ratings.get(away_team, self.elo_config.initial_rating)
        difference = (
            home_rating - away_rating
        ) / self.elo_config.rating_scale
        venue_effect = 0.0 if neutral_venue else self.parameters.home_advantage
        home_rate = math.exp(
            self.parameters.base_log_goal_rate
            + self.parameters.elo_coefficient * difference
            + venue_effect
        )
        away_rate = math.exp(
            self.parameters.base_log_goal_rate
            - self.parameters.elo_coefficient * difference
        )
        return home_rate, away_rate

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
        device: ComputeDevice = "auto",
        strict_teams: bool = True,
    ) -> MatchPrediction:
        if home_team == away_team:
            raise ValueError("A team cannot play itself")
        home_rate, away_rate = self.expected_goals(
            home_team, away_team, neutral_venue, strict_teams=strict_teams
        )
        score_matrix, captured_mass, _ = score_matrix_from_rates(
            home_rate=home_rate,
            away_rate=away_rate,
            max_goals=self.model_config.max_goals,
            device=device,
        )

        home_win = float(np.tril(score_matrix, k=-1).sum())
        draw = float(np.trace(score_matrix))
        away_win = float(np.triu(score_matrix, k=1).sum())
        likely_home, likely_away = np.unravel_index(
            int(np.argmax(score_matrix)), score_matrix.shape
        )

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            expected_home_goals=home_rate,
            expected_away_goals=away_rate,
            home_win_prob=home_win,
            draw_prob=draw,
            away_win_prob=away_win,
            most_likely_score=f"{likely_home}-{likely_away}",
            score_matrix=score_matrix.tolist(),
            captured_probability_mass=captured_mass,
        )

    def predict_many(
        self,
        matches: Sequence[MatchSpec],
        device: ComputeDevice = "auto",
        strict_teams: bool = True,
    ) -> list[MatchPrediction]:
        if not matches:
            raise ValueError("At least one match is required")

        home_rates = []
        away_rates = []
        normalized_matches = []
        for home_team, away_team, neutral_venue in matches:
            if home_team == away_team:
                raise ValueError("A team cannot play itself")
            home_rate, away_rate = self.expected_goals(
                home_team,
                away_team,
                neutral_venue,
                strict_teams=strict_teams,
            )
            home_rates.append(home_rate)
            away_rates.append(away_rate)
            normalized_matches.append((home_team, away_team))

        score_matrices, captured_masses, _ = score_matrices_from_rates(
            home_rates=np.asarray(home_rates, dtype=np.float64),
            away_rates=np.asarray(away_rates, dtype=np.float64),
            max_goals=self.model_config.max_goals,
            device=device,
        )

        predictions = []
        for index, (home_team, away_team) in enumerate(normalized_matches):
            score_matrix = score_matrices[index]
            home_win = float(np.tril(score_matrix, k=-1).sum())
            draw = float(np.trace(score_matrix))
            away_win = float(np.triu(score_matrix, k=1).sum())
            likely_home, likely_away = np.unravel_index(
                int(np.argmax(score_matrix)),
                score_matrix.shape,
            )
            predictions.append(
                MatchPrediction(
                    home_team=home_team,
                    away_team=away_team,
                    expected_home_goals=home_rates[index],
                    expected_away_goals=away_rates[index],
                    home_win_prob=home_win,
                    draw_prob=draw,
                    away_win_prob=away_win,
                    most_likely_score=f"{likely_home}-{likely_away}",
                    score_matrix=score_matrix.tolist(),
                    captured_probability_mass=float(captured_masses[index]),
                )
            )
        return predictions

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        competition_type: str,
        neutral_venue: bool,
    ) -> None:
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before updating ratings")
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

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)

    def to_dict(self) -> dict[str, Any]:
        if self.parameters is None:
            raise RuntimeError("Cannot serialize an unfitted model")
        return {
            "model_version": self.model_version,
            "trained_through": self.trained_through,
            "elo_config": self.elo_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "parameters": asdict(self.parameters),
            "ratings": self.ratings,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "EloPoissonModel":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("model_version") != cls.model_version:
            raise ValueError(
                f"Unsupported model version: {payload.get('model_version')!r}"
            )
        model = cls(
            elo_config=EloConfig.from_dict(payload["elo_config"]),
            model_config=ModelConfig.from_dict(payload["model_config"]),
        )
        model.parameters = PoissonParameters(**payload["parameters"])
        model.ratings = {
            str(team): float(rating) for team, rating in payload["ratings"].items()
        }
        model.trained_through = payload.get("trained_through")
        return model
