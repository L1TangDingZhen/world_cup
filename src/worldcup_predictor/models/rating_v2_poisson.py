from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from worldcup_predictor.compute import ComputeDevice, score_matrix_from_rates
from worldcup_predictor.config import ModelConfig
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.elo_poisson import MatchPrediction
from worldcup_predictor.ratings.v2 import FootballRatingV2, RatingV2Config


@dataclass(frozen=True)
class RatingV2PoissonParameters:
    base_log_goal_rate: float
    rating_coefficient: float
    uncertainty_coefficient: float
    form_coefficient: float
    home_advantage: float
    time_decay_xi: float


class RatingV2PoissonModel:
    model_version = "rating_v2_poisson_v1"

    def __init__(
        self,
        rating_config: RatingV2Config | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.rating_config = rating_config or RatingV2Config()
        self.model_config = model_config or ModelConfig()
        self.parameters: RatingV2PoissonParameters | None = None
        self.ratings: dict[str, float] = {}
        self.uncertainties: dict[str, float] = {}
        self.recent_forms: dict[str, float] = {}
        self.trained_through: str | None = None

    @property
    def is_fitted(self) -> bool:
        return self.parameters is not None and bool(self.ratings)

    def fit(self, matches: pd.DataFrame) -> "RatingV2PoissonModel":
        frame = validate_matches(matches)
        if len(frame) < 3:
            raise ValueError("At least three matches are required to fit the model")

        engine = FootballRatingV2(config=self.rating_config)
        history = engine.process(frame)
        rating_diff = (
            history["home_rating_before"].to_numpy(dtype=float)
            - history["away_rating_before"].to_numpy(dtype=float)
        ) / self.rating_config.rating_scale
        uncertainty_sum = (
            history["home_uncertainty_before"].to_numpy(dtype=float)
            + history["away_uncertainty_before"].to_numpy(dtype=float)
        ) / (2.0 * self.rating_config.initial_uncertainty)
        form_diff = (
            history["home_recent_form_before"].to_numpy(dtype=float)
            - history["away_recent_form_before"].to_numpy(dtype=float)
        )
        home_indicator = (~history["neutral_venue"]).to_numpy(dtype=float)
        home_goals = history["home_goals"].to_numpy(dtype=float)
        away_goals = history["away_goals"].to_numpy(dtype=float)

        latest_date = history["date"].max()
        age_days = (latest_date - history["date"]).dt.days.to_numpy(dtype=float)
        xi = math.log(2.0) / self.model_config.time_decay_half_life_days
        weights = np.exp(-xi * age_days)

        mean_goals = max((home_goals.sum() + away_goals.sum()) / (2 * len(frame)), 0.1)
        initial = np.array([math.log(mean_goals), 0.35, 0.0, 0.10, 0.05], dtype=float)

        def objective(values: np.ndarray) -> float:
            base, rating_coef, uncertainty_coef, form_coef, home_advantage = values
            home_log_rate = np.clip(
                base
                + rating_coef * rating_diff
                + uncertainty_coef * uncertainty_sum
                + form_coef * form_diff
                + home_advantage * home_indicator,
                -5.0,
                5.0,
            )
            away_log_rate = np.clip(
                base
                - rating_coef * rating_diff
                + uncertainty_coef * uncertainty_sum
                - form_coef * form_diff,
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
            bounds=((-3.0, 2.0), (0.0, 3.0), (-1.0, 1.0), (-2.0, 2.0), (-1.0, 1.0)),
        )
        if not result.success:
            raise RuntimeError(f"Rating V2 Poisson fitting failed: {result.message}")

        self.parameters = RatingV2PoissonParameters(
            base_log_goal_rate=float(result.x[0]),
            rating_coefficient=float(result.x[1]),
            uncertainty_coefficient=float(result.x[2]),
            form_coefficient=float(result.x[3]),
            home_advantage=float(result.x[4]),
            time_decay_xi=xi,
        )
        self.ratings = engine.ratings()
        self.uncertainties = engine.uncertainties()
        self.recent_forms = engine.recent_forms()
        self.trained_through = latest_date.date().isoformat()
        return self

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
    ) -> tuple[float, float]:
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before prediction")
        home_rating = self.ratings.get(home_team, self.rating_config.initial_rating)
        away_rating = self.ratings.get(away_team, self.rating_config.initial_rating)
        home_uncertainty = self.uncertainties.get(
            home_team,
            self.rating_config.initial_uncertainty,
        )
        away_uncertainty = self.uncertainties.get(
            away_team,
            self.rating_config.initial_uncertainty,
        )
        home_form = self.recent_forms.get(home_team, 0.0)
        away_form = self.recent_forms.get(away_team, 0.0)

        rating_diff = (home_rating - away_rating) / self.rating_config.rating_scale
        uncertainty_sum = (
            home_uncertainty + away_uncertainty
        ) / (2.0 * self.rating_config.initial_uncertainty)
        form_diff = home_form - away_form
        venue_effect = 0.0 if neutral_venue else self.parameters.home_advantage

        home_rate = math.exp(
            self.parameters.base_log_goal_rate
            + self.parameters.rating_coefficient * rating_diff
            + self.parameters.uncertainty_coefficient * uncertainty_sum
            + self.parameters.form_coefficient * form_diff
            + venue_effect
        )
        away_rate = math.exp(
            self.parameters.base_log_goal_rate
            - self.parameters.rating_coefficient * rating_diff
            + self.parameters.uncertainty_coefficient * uncertainty_sum
            - self.parameters.form_coefficient * form_diff
        )
        return home_rate, away_rate

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
        device: ComputeDevice = "auto",
    ) -> MatchPrediction:
        if home_team == away_team:
            raise ValueError("A team cannot play itself")
        home_rate, away_rate = self.expected_goals(home_team, away_team, neutral_venue)
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
            int(np.argmax(score_matrix)),
            score_matrix.shape,
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

    def update_ratings(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        competition_type: str,
        neutral_venue: bool,
        date: pd.Timestamp | str | None = None,
    ) -> None:
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before updating ratings")
        engine = FootballRatingV2(
            config=self.rating_config,
            states=_states_from_parts(
                self.ratings,
                self.uncertainties,
                self.recent_forms,
            ),
        )
        engine.update(
            home_team=home_team,
            away_team=away_team,
            home_goals=home_goals,
            away_goals=away_goals,
            competition_type=competition_type,
            neutral_venue=neutral_venue,
            date=pd.Timestamp(date) if date is not None else None,
        )
        self.ratings = engine.ratings()
        self.uncertainties = engine.uncertainties()
        self.recent_forms = engine.recent_forms()

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)

    def to_dict(self) -> dict[str, Any]:
        if self.parameters is None:
            raise RuntimeError("Cannot serialize an unfitted model")
        return {
            "model_version": self.model_version,
            "trained_through": self.trained_through,
            "rating_config": self.rating_config.to_dict(),
            "model_config": self.model_config.to_dict(),
            "parameters": asdict(self.parameters),
            "ratings": self.ratings,
            "uncertainties": self.uncertainties,
            "recent_forms": self.recent_forms,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RatingV2PoissonModel":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("model_version") != cls.model_version:
            raise ValueError(f"Unsupported model version: {payload.get('model_version')!r}")
        model = cls(
            rating_config=RatingV2Config.from_dict(payload["rating_config"]),
            model_config=ModelConfig.from_dict(payload["model_config"]),
        )
        model.parameters = RatingV2PoissonParameters(**payload["parameters"])
        model.ratings = {str(team): float(value) for team, value in payload["ratings"].items()}
        model.uncertainties = {
            str(team): float(value) for team, value in payload["uncertainties"].items()
        }
        model.recent_forms = {
            str(team): float(value) for team, value in payload["recent_forms"].items()
        }
        model.trained_through = payload.get("trained_through")
        return model


def _states_from_parts(
    ratings: dict[str, float],
    uncertainties: dict[str, float],
    recent_forms: dict[str, float],
):
    from worldcup_predictor.ratings.v2 import TeamRatingState

    return {
        team: TeamRatingState(
            rating=rating,
            uncertainty=uncertainties.get(team, 120.0),
            recent_form=recent_forms.get(team, 0.0),
        )
        for team, rating in ratings.items()
    }
