from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from worldcup_predictor.models.elo_poisson import MatchPrediction


REQUIRED_PLAYER_COLUMNS = {
    "team",
    "player",
    "attacking_rating",
    "defensive_rating",
    "available",
}


@dataclass(frozen=True)
class TeamPlayerAdjustment:
    team: str
    attack_adjustment: float
    defense_adjustment: float
    available_players: int


class PredictsExpectedGoals(Protocol):
    ratings: dict[str, float]

    def predict(self, home_team: str, away_team: str, neutral_venue: bool = True) -> MatchPrediction:
        ...


def load_players(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return validate_players(frame)


def validate_players(players: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_PLAYER_COLUMNS - set(players.columns)
    if missing:
        raise ValueError(f"Missing player columns: {', '.join(sorted(missing))}")
    frame = players.copy()
    frame["team"] = frame["team"].astype("string").str.strip()
    frame["player"] = frame["player"].astype("string").str.strip()
    frame["attacking_rating"] = pd.to_numeric(frame["attacking_rating"], errors="raise")
    frame["defensive_rating"] = pd.to_numeric(frame["defensive_rating"], errors="raise")
    frame["available"] = frame["available"].map(_parse_bool).astype(bool)
    if frame["team"].eq("").any() or frame["player"].eq("").any():
        raise ValueError("team/player cannot be empty")
    return frame


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def squad_attack_adjustment(players: pd.DataFrame, rating_column: str = "attacking_rating") -> float:
    if players.empty or rating_column not in players:
        return 0.0
    ratings = pd.to_numeric(players[rating_column], errors="coerce").dropna()
    if ratings.empty:
        return 0.0
    return float((ratings.mean() - 70.0) / 100.0)


def aggregate_team_player_features(players: pd.DataFrame) -> dict[str, TeamPlayerAdjustment]:
    frame = validate_players(players)
    available = frame.loc[frame["available"]].copy()
    if available.empty:
        return {}
    rows: dict[str, TeamPlayerAdjustment] = {}
    for team, group in available.groupby("team"):
        attack = float((group["attacking_rating"].mean() - 70.0) / 100.0)
        # Higher defensive rating should reduce opponent goals, hence negative weakness.
        defense = float(-(group["defensive_rating"].mean() - 70.0) / 100.0)
        rows[str(team)] = TeamPlayerAdjustment(
            team=str(team),
            attack_adjustment=attack,
            defense_adjustment=defense,
            available_players=len(group),
        )
    return rows


class PlayerAdjustedPredictor:
    def __init__(
        self,
        base_model: PredictsExpectedGoals,
        adjustments: dict[str, TeamPlayerAdjustment],
        scale: float = 0.15,
    ) -> None:
        self.base_model = base_model
        self.adjustments = adjustments
        self.scale = scale
        self.ratings = base_model.ratings

    def predict(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
    ) -> MatchPrediction:
        prediction = self.base_model.predict(home_team, away_team, neutral_venue)
        home_adj = self.adjustments.get(home_team)
        away_adj = self.adjustments.get(away_team)
        if home_adj is None and away_adj is None:
            return prediction
        home_factor = 1.0
        away_factor = 1.0
        if home_adj is not None:
            home_factor += self.scale * home_adj.attack_adjustment
            away_factor += self.scale * home_adj.defense_adjustment
        if away_adj is not None:
            away_factor += self.scale * away_adj.attack_adjustment
            home_factor += self.scale * away_adj.defense_adjustment
        home_factor = max(home_factor, 0.25)
        away_factor = max(away_factor, 0.25)
        from worldcup_predictor.models.elo_poisson import EloPoissonModel

        # Reuse a tiny temporary Poisson matrix through expected-rate scaling.
        temp = EloPoissonModel()
        temp.ratings = {"home": 1500.0, "away": 1500.0}
        temp.parameters = type(
            "Params",
            (),
            {
                "base_log_goal_rate": 0.0,
                "elo_coefficient": 0.0,
                "home_advantage": 0.0,
            },
        )()
        temp.model_config = self.base_model.model_config  # type: ignore[attr-defined]
        return _prediction_from_rates(
            prediction,
            prediction.expected_home_goals * home_factor,
            prediction.expected_away_goals * away_factor,
        )


def _prediction_from_rates(original: MatchPrediction, home_rate: float, away_rate: float) -> MatchPrediction:
    import math
    import numpy as np

    max_goals = len(original.score_matrix) - 1
    home = np.empty(max_goals + 1)
    away = np.empty(max_goals + 1)
    home[0] = math.exp(-home_rate)
    away[0] = math.exp(-away_rate)
    for goals in range(1, max_goals + 1):
        home[goals] = home[goals - 1] * home_rate / goals
        away[goals] = away[goals - 1] * away_rate / goals
    matrix = np.outer(home, away)
    captured = float(matrix.sum())
    matrix = matrix / captured
    home_win = float(np.tril(matrix, k=-1).sum())
    draw = float(np.trace(matrix))
    away_win = float(np.triu(matrix, k=1).sum())
    likely_home, likely_away = np.unravel_index(int(np.argmax(matrix)), matrix.shape)
    return MatchPrediction(
        home_team=original.home_team,
        away_team=original.away_team,
        expected_home_goals=home_rate,
        expected_away_goals=away_rate,
        home_win_prob=home_win,
        draw_prob=draw,
        away_win_prob=away_win,
        most_likely_score=f"{likely_home}-{likely_away}",
        score_matrix=matrix.tolist(),
        captured_probability_mass=captured,
    )
