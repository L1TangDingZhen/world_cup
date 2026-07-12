from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from worldcup_predictor.config import ModelConfig
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.models.elo_poisson import MatchPrediction


@dataclass(frozen=True)
class BayesianPredictionInterval:
    home_win_low: float
    home_win_high: float
    draw_low: float
    draw_high: float
    away_win_low: float
    away_win_high: float


class BayesianHierarchicalModel:
    """Empirical-Bayes hierarchical Poisson model.

    This is a working lightweight Step 11 implementation: team attack/defense
    effects are MAP estimates under Gaussian shrinkage priors, and predictions
    average over posterior-like normal draws around those MAP effects. A full
    PyMC implementation can replace this class without changing simulator/API
    contracts because it exposes the same `predict()` shape.
    """

    model_version = "bayesian_hierarchical_v1"

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        prior_sd: float = 0.45,
        posterior_draws: int = 200,
        random_seed: int = 42,
        max_iterations: int = 500,
    ) -> None:
        self.model_config = model_config or ModelConfig()
        self.prior_sd = prior_sd
        self.posterior_draws = posterior_draws
        self.rng = np.random.default_rng(random_seed)
        self.max_iterations = max_iterations
        self.teams: list[str] = []
        self.intercept = 0.0
        self.home_advantage = 0.0
        self.attack: dict[str, float] = {}
        self.defense: dict[str, float] = {}
        self.posterior_sd = prior_sd / 3.0
        self.trained_through: str | None = None
        self.optimization_success: bool | None = None
        self.optimization_message: str | None = None

    @property
    def ratings(self) -> dict[str, float]:
        return {
            team: 1500.0 + 200.0 * (self.attack.get(team, 0.0) - self.defense.get(team, 0.0))
            for team in self.teams
        }

    def fit(self, matches: pd.DataFrame) -> "BayesianHierarchicalModel":
        frame = validate_matches(matches)
        if len(frame) < 3:
            raise ValueError("At least three matches are required")
        self.teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
        team_index = {team: index for index, team in enumerate(self.teams)}
        n_teams = len(self.teams)
        home_idx = frame["home_team"].map(team_index).to_numpy(dtype=int)
        away_idx = frame["away_team"].map(team_index).to_numpy(dtype=int)
        home_goals = frame["home_goals"].to_numpy(dtype=float)
        away_goals = frame["away_goals"].to_numpy(dtype=float)
        home_indicator = (~frame["neutral_venue"]).to_numpy(dtype=float)
        mean_goals = max((home_goals.sum() + away_goals.sum()) / (2 * len(frame)), 0.1)
        initial = np.zeros(2 + 2 * n_teams, dtype=float)
        initial[0] = math.log(mean_goals)
        initial[1] = 0.10

        def unpack(values: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
            attack = values[2 : 2 + n_teams]
            defense = values[2 + n_teams :]
            return values[0], values[1], attack - attack.mean(), defense - defense.mean()

        def objective(values: np.ndarray) -> float:
            intercept, home_advantage, attack, defense = unpack(values)
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
            nll = np.sum(home_rate - home_goals * home_log_rate + away_rate - away_goals * away_log_rate)
            prior_penalty = 0.5 * (
                np.sum((attack / self.prior_sd) ** 2)
                + np.sum((defense / self.prior_sd) ** 2)
                + (home_advantage / self.prior_sd) ** 2
            )
            return float((nll + prior_penalty) / len(frame))

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(-3, 2), (-1, 1)] + [(-2, 2)] * (2 * n_teams),
            options={"maxiter": self.max_iterations},
        )
        self.optimization_success = bool(result.success)
        self.optimization_message = str(result.message)
        if (not result.success) and (not np.isfinite(result.fun)):
            raise RuntimeError(f"Bayesian MAP fitting failed: {result.message}")
        intercept, home_advantage, attack, defense = unpack(result.x)
        self.intercept = float(intercept)
        self.home_advantage = float(home_advantage)
        self.attack = {team: float(attack[index]) for team, index in team_index.items()}
        self.defense = {team: float(defense[index]) for team, index in team_index.items()}
        self.posterior_sd = min(self.prior_sd / 2.0, 1.0 / math.sqrt(max(len(frame) / max(n_teams, 1), 1.0)))
        self.trained_through = frame["date"].max().date().isoformat()
        return self

    def _score_matrix(self, home_rate: float, away_rate: float) -> np.ndarray:
        max_goals = self.model_config.max_goals
        home = np.empty(max_goals + 1)
        away = np.empty(max_goals + 1)
        home[0] = math.exp(-home_rate)
        away[0] = math.exp(-away_rate)
        for goals in range(1, max_goals + 1):
            home[goals] = home[goals - 1] * home_rate / goals
            away[goals] = away[goals - 1] * away_rate / goals
        matrix = np.outer(home, away)
        return matrix / matrix.sum()

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
    ) -> MatchPrediction:
        if not self.teams:
            raise RuntimeError("Model must be fitted before prediction")
        venue = 0.0 if neutral_venue else self.home_advantage
        matrices = []
        home_rates = []
        away_rates = []
        draws = max(1, self.posterior_draws)
        for _ in range(draws):
            home_attack = self.rng.normal(self.attack.get(home_team, 0.0), self.posterior_sd)
            away_attack = self.rng.normal(self.attack.get(away_team, 0.0), self.posterior_sd)
            home_defense = self.rng.normal(self.defense.get(home_team, 0.0), self.posterior_sd)
            away_defense = self.rng.normal(self.defense.get(away_team, 0.0), self.posterior_sd)
            home_rate = math.exp(self.intercept + home_attack + away_defense + venue)
            away_rate = math.exp(self.intercept + away_attack + home_defense)
            home_rates.append(home_rate)
            away_rates.append(away_rate)
            matrices.append(self._score_matrix(home_rate, away_rate))
        matrix = np.mean(matrices, axis=0)
        matrix = matrix / matrix.sum()
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        likely_home, likely_away = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            expected_home_goals=float(np.mean(home_rates)),
            expected_away_goals=float(np.mean(away_rates)),
            home_win_prob=home_win,
            draw_prob=draw,
            away_win_prob=away_win,
            most_likely_score=f"{likely_home}-{likely_away}",
            score_matrix=matrix.tolist(),
            captured_probability_mass=1.0,
        )

    def prediction_interval(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
        draws: int = 500,
    ) -> BayesianPredictionInterval:
        old_draws = self.posterior_draws
        self.posterior_draws = 1
        samples = []
        for _ in range(draws):
            p = self.predict(home_team, away_team, neutral_venue)
            samples.append([p.home_win_prob, p.draw_prob, p.away_win_prob])
        self.posterior_draws = old_draws
        values = np.asarray(samples)
        low = np.quantile(values, 0.05, axis=0)
        high = np.quantile(values, 0.95, axis=0)
        return BayesianPredictionInterval(
            home_win_low=float(low[0]),
            home_win_high=float(high[0]),
            draw_low=float(low[1]),
            draw_high=float(high[1]),
            away_win_low=float(low[2]),
            away_win_high=float(high[2]),
        )

    def to_dict(self) -> dict[str, Any]:
        if not self.teams:
            raise RuntimeError("Cannot serialize an unfitted model")
        return {
            "model_version": self.model_version,
            "model_config": self.model_config.to_dict(),
            "prior_sd": self.prior_sd,
            "posterior_draws": self.posterior_draws,
            "max_iterations": self.max_iterations,
            "teams": self.teams,
            "intercept": self.intercept,
            "home_advantage": self.home_advantage,
            "attack": self.attack,
            "defense": self.defense,
            "posterior_sd": self.posterior_sd,
            "trained_through": self.trained_through,
            "optimization_success": self.optimization_success,
            "optimization_message": self.optimization_message,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BayesianHierarchicalModel":
        from worldcup_predictor.config import ModelConfig

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("model_version") != cls.model_version:
            raise ValueError(f"Unsupported model version: {payload.get('model_version')!r}")
        model = cls(
            model_config=ModelConfig.from_dict(payload["model_config"]),
            prior_sd=float(payload["prior_sd"]),
            posterior_draws=int(payload["posterior_draws"]),
            max_iterations=int(payload.get("max_iterations", 500)),
        )
        model.teams = [str(team) for team in payload["teams"]]
        model.intercept = float(payload["intercept"])
        model.home_advantage = float(payload["home_advantage"])
        model.attack = {str(team): float(value) for team, value in payload["attack"].items()}
        model.defense = {str(team): float(value) for team, value in payload["defense"].items()}
        model.posterior_sd = float(payload["posterior_sd"])
        model.trained_through = payload.get("trained_through")
        model.optimization_success = payload.get("optimization_success")
        model.optimization_message = payload.get("optimization_message")
        return model


class PyMCBayesianHierarchicalModel:
    """PyMC hierarchical Poisson model with posterior predictive score matrices."""

    model_version = "pymc_hierarchical_poisson_v1"

    def __init__(self, model_config: ModelConfig | None = None, random_seed: int = 42) -> None:
        try:
            os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
            os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
            if "PYTENSOR_FLAGS" not in os.environ:
                os.environ["PYTENSOR_FLAGS"] = "compiledir=/tmp/pytensor"
            import pymc as pm
        except ImportError as exc:
            raise ImportError("Install pymc to use PyMCBayesianHierarchicalModel.") from exc
        self.pm = pm
        self.model_config = model_config or ModelConfig()
        self.random_seed = random_seed
        self.teams: list[str] = []
        self.idata = None
        self.trained_through: str | None = None

    @property
    def ratings(self) -> dict[str, float]:
        if self.idata is None:
            return {}
        attack = self.idata.posterior["attack"].mean(("chain", "draw")).values
        defense = self.idata.posterior["defense"].mean(("chain", "draw")).values
        return {
            team: float(1500.0 + 200.0 * (attack[index] - defense[index]))
            for index, team in enumerate(self.teams)
        }

    def fit(
        self,
        matches: pd.DataFrame,
        draws: int = 300,
        tune: int = 300,
        chains: int = 2,
    ) -> "PyMCBayesianHierarchicalModel":
        frame = validate_matches(matches)
        self.teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
        team_index = {team: index for index, team in enumerate(self.teams)}
        home_idx = frame["home_team"].map(team_index).to_numpy(dtype=int)
        away_idx = frame["away_team"].map(team_index).to_numpy(dtype=int)
        home_goals = frame["home_goals"].to_numpy(dtype=int)
        away_goals = frame["away_goals"].to_numpy(dtype=int)
        home_indicator = (~frame["neutral_venue"]).to_numpy(dtype=float)
        mean_goals = max((home_goals.sum() + away_goals.sum()) / (2 * len(frame)), 0.1)
        pm = self.pm
        coords = {"team": self.teams, "match": np.arange(len(frame))}
        with pm.Model(coords=coords) as model:
            sigma_attack = pm.HalfNormal("sigma_attack", sigma=0.5)
            sigma_defense = pm.HalfNormal("sigma_defense", sigma=0.5)
            attack_raw = pm.Normal("attack_raw", 0.0, sigma_attack, dims="team")
            defense_raw = pm.Normal("defense_raw", 0.0, sigma_defense, dims="team")
            attack = pm.Deterministic("attack", attack_raw - pm.math.mean(attack_raw), dims="team")
            defense = pm.Deterministic("defense", defense_raw - pm.math.mean(defense_raw), dims="team")
            intercept = pm.Normal("intercept", mu=math.log(mean_goals), sigma=1.0)
            home_advantage = pm.Normal("home_advantage", mu=0.0, sigma=0.35)
            home_rate = pm.math.exp(intercept + attack[home_idx] + defense[away_idx] + home_advantage * home_indicator)
            away_rate = pm.math.exp(intercept + attack[away_idx] + defense[home_idx])
            pm.Poisson("home_goals", mu=home_rate, observed=home_goals, dims="match")
            pm.Poisson("away_goals", mu=away_rate, observed=away_goals, dims="match")
            self.idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                progressbar=False,
                random_seed=self.random_seed,
                target_accept=0.9,
            )
        self.trained_through = frame["date"].max().date().isoformat()
        return self

    def predict(self, home_team: str, away_team: str, neutral_venue: bool = True) -> MatchPrediction:
        if self.idata is None:
            raise RuntimeError("Model must be fitted before prediction")
        home_index = self.teams.index(home_team) if home_team in self.teams else None
        away_index = self.teams.index(away_team) if away_team in self.teams else None
        posterior = self.idata.posterior.stack(sample=("chain", "draw"))
        intercept = posterior["intercept"].values
        home_adv = posterior["home_advantage"].values
        attack = posterior["attack"].values
        defense = posterior["defense"].values
        home_attack = attack[home_index] if home_index is not None else np.zeros_like(intercept)
        away_attack = attack[away_index] if away_index is not None else np.zeros_like(intercept)
        home_defense = defense[home_index] if home_index is not None else np.zeros_like(intercept)
        away_defense = defense[away_index] if away_index is not None else np.zeros_like(intercept)
        home_rates = np.exp(intercept + home_attack + away_defense + (0.0 if neutral_venue else home_adv))
        away_rates = np.exp(intercept + away_attack + home_defense)
        max_goals = self.model_config.max_goals
        matrices = []
        for home_rate, away_rate in zip(home_rates, away_rates, strict=True):
            home_probs = np.array([math.exp(-home_rate) * home_rate**goals / math.factorial(goals) for goals in range(max_goals + 1)])
            away_probs = np.array([math.exp(-away_rate) * away_rate**goals / math.factorial(goals) for goals in range(max_goals + 1)])
            matrix = np.outer(home_probs, away_probs)
            matrices.append(matrix / matrix.sum())
        matrix = np.mean(matrices, axis=0)
        matrix = matrix / matrix.sum()
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        likely_home, likely_away = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            expected_home_goals=float(np.mean(home_rates)),
            expected_away_goals=float(np.mean(away_rates)),
            home_win_prob=home_win,
            draw_prob=draw,
            away_win_prob=away_win,
            most_likely_score=f"{likely_home}-{likely_away}",
            score_matrix=matrix.tolist(),
            captured_probability_mass=1.0,
        )

    def save(self, path: str | Path) -> None:
        if self.idata is None:
            raise RuntimeError("Cannot serialize an unfitted model")
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self.idata.to_netcdf(output)
        output.with_suffix(output.suffix + ".metadata.json").write_text(
            json.dumps(
                {
                    "model_version": self.model_version,
                    "model_config": self.model_config.to_dict(),
                    "teams": self.teams,
                    "trained_through": self.trained_through,
                    "random_seed": self.random_seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
