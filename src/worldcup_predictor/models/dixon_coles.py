from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from worldcup_predictor.config import ModelConfig
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.elo_poisson import MatchPrediction


@dataclass(frozen=True)
class DixonColesParameters:
    intercept: float
    home_advantage: float
    rho: float
    time_decay_xi: float


def dixon_coles_tau(
    home_goals: int,
    away_goals: int,
    home_rate: float,
    away_rate: float,
    rho: float,
) -> float:
    if home_goals == 0 and away_goals == 0:
        return max(1.0 - home_rate * away_rate * rho, 1e-12)
    if home_goals == 0 and away_goals == 1:
        return max(1.0 + home_rate * rho, 1e-12)
    if home_goals == 1 and away_goals == 0:
        return max(1.0 + away_rate * rho, 1e-12)
    if home_goals == 1 and away_goals == 1:
        return max(1.0 - rho, 1e-12)
    return 1.0


def _poisson_log_pmf(goals: np.ndarray, rates: np.ndarray) -> np.ndarray:
    return goals * np.log(rates) - rates - np.vectorize(math.lgamma)(goals + 1.0)


def _poisson_probabilities(rate: float, max_goals: int) -> np.ndarray:
    probabilities = np.empty(max_goals + 1, dtype=float)
    probabilities[0] = math.exp(-rate)
    for goals in range(1, max_goals + 1):
        probabilities[goals] = probabilities[goals - 1] * rate / goals
    return probabilities


class DixonColesModel:
    model_version = "dixon_coles_v1"

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        max_iterations: int = 5_000,
        max_function_evaluations: int | None = None,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.max_iterations = max_iterations
        self.max_function_evaluations = (
            max_function_evaluations
            if max_function_evaluations is not None
            else max(50_000, max_iterations * 20)
        )
        self.parameters: DixonColesParameters | None = None
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.teams: list[str] = []
        self.trained_through: str | None = None
        self.optimization_success: bool | None = None
        self.optimization_message: str | None = None
        self.optimization_iterations: int | None = None
        self.optimization_function_evaluations: int | None = None

    @property
    def ratings(self) -> dict[str, float]:
        # Simulation penalty shootouts only need a monotonic strength proxy.
        return {
            team: 1500.0 + 200.0 * (self.attack.get(team, 0.0) - self.defense.get(team, 0.0))
            for team in self.teams
        }

    @property
    def is_fitted(self) -> bool:
        return self.parameters is not None and bool(self.teams)

    def fit(self, matches: pd.DataFrame) -> "DixonColesModel":
        frame = validate_matches(matches)
        if len(frame) < 3:
            raise ValueError("At least three matches are required to fit Dixon-Coles")

        self.teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
        team_index = {team: index for index, team in enumerate(self.teams)}
        n_teams = len(self.teams)
        home_idx = frame["home_team"].map(team_index).to_numpy(dtype=int)
        away_idx = frame["away_team"].map(team_index).to_numpy(dtype=int)
        home_goals = frame["home_goals"].to_numpy(dtype=float)
        away_goals = frame["away_goals"].to_numpy(dtype=float)
        home_indicator = (~frame["neutral_venue"]).to_numpy(dtype=float)
        latest_date = frame["date"].max()
        age_days = (latest_date - frame["date"]).dt.days.to_numpy(dtype=float)
        xi = math.log(2.0) / self.model_config.time_decay_half_life_days
        weights = np.exp(-xi * age_days)
        mean_goals = max((home_goals.sum() + away_goals.sum()) / (2 * len(frame)), 0.1)

        # Use N-1 free attack and defense parameters.  The final value is the
        # negative sum of the others, imposing sum(attack)=sum(defense)=0 and
        # removing the likelihood's otherwise flat identifiability directions.
        initial = np.zeros(3 + 2 * (n_teams - 1), dtype=float)
        initial[0] = math.log(mean_goals)
        initial[1] = 0.10
        initial[2] = -0.05

        def unpack(values: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
            intercept = values[0]
            home_advantage = values[1]
            rho = values[2]
            attack_free = values[3 : 3 + n_teams - 1]
            defense_free = values[3 + n_teams - 1 :]
            attack = np.concatenate((attack_free, [-attack_free.sum()]))
            defense = np.concatenate((defense_free, [-defense_free.sum()]))
            return (
                intercept,
                home_advantage,
                rho,
                attack,
                defense,
            )

        def objective_and_gradient(values: np.ndarray) -> tuple[float, np.ndarray]:
            intercept, home_advantage, rho, attack, defense = unpack(values)
            home_log_rate = np.clip(
                intercept + attack[home_idx] + defense[away_idx] + home_advantage * home_indicator,
                -5.0,
                5.0,
            )
            away_log_rate = np.clip(
                intercept + attack[away_idx] + defense[home_idx],
                -5.0,
                5.0,
            )
            home_rate = np.exp(home_log_rate)
            away_rate = np.exp(away_log_rate)
            tau = np.ones(len(frame), dtype=float)
            tau_home_log_gradient = np.zeros(len(frame), dtype=float)
            tau_away_log_gradient = np.zeros(len(frame), dtype=float)
            tau_rho_gradient = np.zeros(len(frame), dtype=float)

            is_00 = (home_goals == 0) & (away_goals == 0)
            is_01 = (home_goals == 0) & (away_goals == 1)
            is_10 = (home_goals == 1) & (away_goals == 0)
            is_11 = (home_goals == 1) & (away_goals == 1)

            def apply_tau(
                mask: np.ndarray,
                raw_tau: np.ndarray | float,
                home_derivative: np.ndarray | float,
                away_derivative: np.ndarray | float,
                rho_derivative: np.ndarray | float,
            ) -> None:
                if not mask.any():
                    return
                raw = np.asarray(raw_tau, dtype=float)
                bounded = np.maximum(raw, 1e-12)
                tau[mask] = bounded
                active = raw > 1e-12
                tau_home_log_gradient[mask] = np.where(
                    active,
                    np.asarray(home_derivative, dtype=float) / bounded,
                    0.0,
                )
                tau_away_log_gradient[mask] = np.where(
                    active,
                    np.asarray(away_derivative, dtype=float) / bounded,
                    0.0,
                )
                tau_rho_gradient[mask] = np.where(
                    active,
                    np.asarray(rho_derivative, dtype=float) / bounded,
                    0.0,
                )

            apply_tau(
                is_00,
                1.0 - home_rate[is_00] * away_rate[is_00] * rho,
                -home_rate[is_00] * away_rate[is_00] * rho,
                -home_rate[is_00] * away_rate[is_00] * rho,
                -home_rate[is_00] * away_rate[is_00],
            )
            apply_tau(
                is_01,
                1.0 + home_rate[is_01] * rho,
                home_rate[is_01] * rho,
                0.0,
                home_rate[is_01],
            )
            apply_tau(
                is_10,
                1.0 + away_rate[is_10] * rho,
                0.0,
                away_rate[is_10] * rho,
                away_rate[is_10],
            )
            apply_tau(is_11, 1.0 - rho, 0.0, 0.0, -1.0)

            log_likelihood = (
                _poisson_log_pmf(home_goals, home_rate)
                + _poisson_log_pmf(away_goals, away_rate)
                + np.log(tau)
            )
            normalizer = weights.sum()
            objective = float(-(weights * log_likelihood).sum() / normalizer)

            home_score = home_goals - home_rate + tau_home_log_gradient
            away_score = away_goals - away_rate + tau_away_log_gradient
            weighted_home_score = weights * home_score
            weighted_away_score = weights * away_score
            attack_score = (
                np.bincount(home_idx, weights=weighted_home_score, minlength=n_teams)
                + np.bincount(away_idx, weights=weighted_away_score, minlength=n_teams)
            )
            defense_score = (
                np.bincount(away_idx, weights=weighted_home_score, minlength=n_teams)
                + np.bincount(home_idx, weights=weighted_away_score, minlength=n_teams)
            )
            gradient = np.empty_like(values)
            gradient[0] = -float((weighted_home_score + weighted_away_score).sum() / normalizer)
            gradient[1] = -float((weighted_home_score * home_indicator).sum() / normalizer)
            gradient[2] = -float((weights * tau_rho_gradient).sum() / normalizer)
            gradient[3 : 3 + n_teams - 1] = -(
                attack_score[:-1] - attack_score[-1]
            ) / normalizer
            gradient[3 + n_teams - 1 :] = -(
                defense_score[:-1] - defense_score[-1]
            ) / normalizer
            return objective, gradient

        bounds = [(-3.0, 2.0), (-1.0, 1.0), (-0.3, 0.3)] + [(-2.5, 2.5)] * (2 * (n_teams - 1))
        result = minimize(
            objective_and_gradient,
            initial,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": self.max_iterations,
                "maxfun": self.max_function_evaluations,
                "ftol": 1e-10,
                "gtol": 1e-6,
            },
        )
        self.optimization_success = bool(result.success)
        self.optimization_message = str(result.message)
        self.optimization_iterations = int(result.nit)
        self.optimization_function_evaluations = int(result.nfev)
        if (not result.success) and (not np.isfinite(result.fun)):
            raise RuntimeError(f"Dixon-Coles fitting failed: {result.message}")

        intercept, home_advantage, rho, attack, defense = unpack(result.x)
        self.parameters = DixonColesParameters(
            intercept=float(intercept),
            home_advantage=float(home_advantage),
            rho=float(rho),
            time_decay_xi=xi,
        )
        self.attack = {team: float(attack[index]) for team, index in team_index.items()}
        self.defense = {team: float(defense[index]) for team, index in team_index.items()}
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
        venue = 0.0 if neutral_venue else self.parameters.home_advantage
        home_rate = math.exp(
            self.parameters.intercept
            + self.attack.get(home_team, 0.0)
            + self.defense.get(away_team, 0.0)
            + venue
        )
        away_rate = math.exp(
            self.parameters.intercept
            + self.attack.get(away_team, 0.0)
            + self.defense.get(home_team, 0.0)
        )
        return home_rate, away_rate

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
    ) -> MatchPrediction:
        if home_team == away_team:
            raise ValueError("A team cannot play itself")
        if self.parameters is None:
            raise RuntimeError("Model must be fitted before prediction")

        home_rate, away_rate = self.expected_goals(home_team, away_team, neutral_venue)
        matrix = np.outer(
            _poisson_probabilities(home_rate, self.model_config.max_goals),
            _poisson_probabilities(away_rate, self.model_config.max_goals),
        )
        captured_mass = float(matrix.sum())
        matrix = matrix / captured_mass
        for home_goals in range(min(2, matrix.shape[0])):
            for away_goals in range(min(2, matrix.shape[1])):
                matrix[home_goals, away_goals] *= dixon_coles_tau(
                    home_goals,
                    away_goals,
                    home_rate,
                    away_rate,
                    self.parameters.rho,
                )
        matrix = matrix / matrix.sum()
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        likely_home, likely_away = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            expected_home_goals=home_rate,
            expected_away_goals=away_rate,
            home_win_prob=home_win,
            draw_prob=draw,
            away_win_prob=away_win,
            most_likely_score=f"{likely_home}-{likely_away}",
            score_matrix=matrix.tolist(),
            captured_probability_mass=captured_mass,
        )

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)

    def to_dict(self) -> dict[str, Any]:
        if self.parameters is None:
            raise RuntimeError("Cannot serialize an unfitted model")
        return {
            "model_version": self.model_version,
            "trained_through": self.trained_through,
            "optimization_success": self.optimization_success,
            "optimization_message": self.optimization_message,
            "optimization_iterations": self.optimization_iterations,
            "optimization_function_evaluations": self.optimization_function_evaluations,
            "model_config": self.model_config.to_dict(),
            "max_iterations": self.max_iterations,
            "max_function_evaluations": self.max_function_evaluations,
            "parameters": asdict(self.parameters),
            "teams": self.teams,
            "attack": self.attack,
            "defense": self.defense,
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DixonColesModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_version") != cls.model_version:
            raise ValueError(f"Unsupported model version: {payload.get('model_version')!r}")
        model = cls(
            model_config=ModelConfig.from_dict(payload["model_config"]),
            max_iterations=int(payload.get("max_iterations", 5_000)),
            max_function_evaluations=payload.get("max_function_evaluations"),
        )
        model.parameters = DixonColesParameters(**payload["parameters"])
        model.teams = [str(team) for team in payload["teams"]]
        model.attack = {str(team): float(value) for team, value in payload["attack"].items()}
        model.defense = {str(team): float(value) for team, value in payload["defense"].items()}
        model.trained_through = payload.get("trained_through")
        model.optimization_success = payload.get("optimization_success")
        model.optimization_message = payload.get("optimization_message")
        model.optimization_iterations = payload.get("optimization_iterations")
        model.optimization_function_evaluations = payload.get(
            "optimization_function_evaluations"
        )
        return model
