from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping

import pandas as pd

from worldcup_predictor.config import EloConfig


def normalize_competition(value: str) -> str:
    # Strip accents first so names like "Copa América" match their patterns.
    decomposed = unicodedata.normalize("NFKD", value.strip().lower())
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^a-z0-9]+", "_", ascii_only).strip("_")
    if "world_cup" in normalized and any(
        word in normalized for word in ("qualif", "prelim")
    ):
        return "world_cup_qualifier"
    if normalized in {"fifa_world_cup", "world_cup"}:
        return "world_cup"
    if "friendly" in normalized:
        return "friendly"
    if "nations_league" in normalized:
        return "nations_league"
    if any(word in normalized for word in ("qualif", "prelim")):
        return "continental_qualifier"
    if any(
        word in normalized
        for word in (
            "asian_cup",
            "africa_cup",
            "african_cup",
            "copa_america",
            "cup_of_nations",
            "euro",
            "gold_cup",
            "nations_cup",
        )
    ):
        return "continental_championship"
    return normalized


class WorldFootballElo:
    def __init__(
        self,
        config: EloConfig | None = None,
        ratings: Mapping[str, float] | None = None,
    ) -> None:
        self.config = config or EloConfig()
        self.ratings = dict(ratings or {})

    def rating_for(self, team: str) -> float:
        return self.ratings.get(team, self.config.initial_rating)

    def expected_home_score(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool,
    ) -> float:
        home_rating = self.rating_for(home_team)
        away_rating = self.rating_for(away_team)
        if not neutral_venue:
            home_rating += self.config.home_advantage_points
        exponent = (away_rating - home_rating) / self.config.rating_scale
        return 1.0 / (1.0 + 10.0**exponent)

    def k_factor(self, competition_type: str) -> float:
        key = normalize_competition(competition_type)
        return self.config.competition_k_factors.get(
            key, self.config.default_k_factor
        )

    @staticmethod
    def goal_difference_multiplier(goal_difference: int) -> float:
        if goal_difference < 0:
            raise ValueError("goal_difference cannot be negative")
        if goal_difference <= 1:
            return 1.0
        if goal_difference == 2:
            return 1.5
        return (11.0 + goal_difference) / 8.0

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        competition_type: str,
        neutral_venue: bool,
    ) -> tuple[float, float]:
        home_before = self.rating_for(home_team)
        away_before = self.rating_for(away_team)
        expected_home = self.expected_home_score(
            home_team, away_team, neutral_venue
        )

        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0

        multiplier = self.goal_difference_multiplier(abs(home_goals - away_goals))
        change = (
            self.k_factor(competition_type)
            * multiplier
            * (actual_home - expected_home)
        )
        self.ratings[home_team] = home_before + change
        self.ratings[away_team] = away_before - change
        return self.ratings[home_team], self.ratings[away_team]

    def process(self, matches: pd.DataFrame) -> pd.DataFrame:
        history: list[dict[str, object]] = []
        for match in matches.itertuples(index=False):
            home_before = self.rating_for(match.home_team)
            away_before = self.rating_for(match.away_team)
            expected_home = self.expected_home_score(
                match.home_team,
                match.away_team,
                bool(match.neutral_venue),
            )
            home_after, away_after = self.update(
                home_team=match.home_team,
                away_team=match.away_team,
                home_goals=int(match.home_goals),
                away_goals=int(match.away_goals),
                competition_type=match.competition_type,
                neutral_venue=bool(match.neutral_venue),
            )
            history.append(
                {
                    "date": match.date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "home_goals": int(match.home_goals),
                    "away_goals": int(match.away_goals),
                    "competition_type": match.competition_type,
                    "neutral_venue": bool(match.neutral_venue),
                    "home_elo_before": home_before,
                    "away_elo_before": away_before,
                    "expected_home_result": expected_home,
                    "home_elo_after": home_after,
                    "away_elo_after": away_after,
                }
            )
        return pd.DataFrame(history)

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda item: item[1], reverse=True)

