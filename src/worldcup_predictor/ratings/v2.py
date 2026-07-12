from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping

import pandas as pd

from worldcup_predictor.ratings.elo import normalize_competition


@dataclass(frozen=True)
class RatingV2Config:
    initial_rating: float = 1500.0
    rating_scale: float = 400.0
    initial_uncertainty: float = 120.0
    min_uncertainty: float = 35.0
    max_uncertainty: float = 220.0
    uncertainty_decay_per_match: float = 0.92
    uncertainty_growth_per_year: float = 35.0
    uncertainty_k_boost: float = 0.35
    home_advantage_points: float = 85.0
    host_advantage_points: float = 35.0
    default_k_factor: float = 18.0
    recent_half_life_days: float = 540.0
    recent_form_scale: float = 60.0
    competition_k_factors: dict[str, float] = field(
        default_factory=lambda: {
            "world_cup": 60.0,
            "continental_championship": 48.0,
            "world_cup_qualifier": 38.0,
            "continental_qualifier": 34.0,
            "nations_league": 28.0,
            "friendly": 14.0,
        }
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> "RatingV2Config":
        return cls(**values)


@dataclass
class TeamRatingState:
    rating: float
    uncertainty: float
    recent_form: float = 0.0
    last_date: pd.Timestamp | None = None


class FootballRatingV2:
    """Football-specific rating engine with uncertainty and recent form.

    This is intentionally independent from ``WorldFootballElo`` so it can be
    evaluated without changing the current production Elo path.
    """

    def __init__(
        self,
        config: RatingV2Config | None = None,
        states: Mapping[str, TeamRatingState] | None = None,
    ) -> None:
        self.config = config or RatingV2Config()
        self.states = dict(states or {})

    def state_for(self, team: str, date: pd.Timestamp | None = None) -> TeamRatingState:
        state = self.states.get(team)
        if state is None:
            return TeamRatingState(
                rating=self.config.initial_rating,
                uncertainty=self.config.initial_uncertainty,
                last_date=date,
            )
        if date is None or state.last_date is None:
            return TeamRatingState(
                rating=state.rating,
                uncertainty=state.uncertainty,
                recent_form=state.recent_form,
                last_date=state.last_date,
            )
        return self._aged_state(state, date)

    def expected_home_score(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool,
        date: pd.Timestamp | None = None,
    ) -> float:
        home_state = self.state_for(home_team, date)
        away_state = self.state_for(away_team, date)
        home_rating = self.effective_rating(home_team, neutral_venue, True, date)
        away_rating = self.effective_rating(away_team, True, False, date)
        uncertainty_penalty = 0.10 * (
            home_state.uncertainty - away_state.uncertainty
        )
        exponent = (
            away_rating - home_rating + uncertainty_penalty
        ) / self.config.rating_scale
        return 1.0 / (1.0 + 10.0**exponent)

    def effective_rating(
        self,
        team: str,
        neutral_venue: bool,
        is_home_team: bool,
        date: pd.Timestamp | None = None,
    ) -> float:
        state = self.state_for(team, date)
        rating = state.rating + self.config.recent_form_scale * state.recent_form
        if is_home_team and not neutral_venue:
            rating += self.config.home_advantage_points
        return rating

    def k_factor(self, competition_type: str) -> float:
        key = normalize_competition(competition_type)
        return self.config.competition_k_factors.get(
            key,
            self.config.default_k_factor,
        )

    @staticmethod
    def goal_difference_multiplier(goal_difference: int) -> float:
        if goal_difference < 0:
            raise ValueError("goal_difference cannot be negative")
        if goal_difference <= 1:
            return 1.0
        return math.log1p(goal_difference) / math.log(2.0)

    def update(
        self,
        home_team: str,
        away_team: str,
        home_goals: int,
        away_goals: int,
        competition_type: str,
        neutral_venue: bool,
        date: pd.Timestamp | None = None,
    ) -> tuple[float, float]:
        date = pd.Timestamp(date) if date is not None else None
        home_before = self.state_for(home_team, date)
        away_before = self.state_for(away_team, date)
        expected_home = self.expected_home_score(
            home_team,
            away_team,
            neutral_venue,
            date,
        )
        actual_home = _actual_home_result(home_goals, away_goals)
        multiplier = self.goal_difference_multiplier(abs(home_goals - away_goals))
        uncertainty_factor = 1.0 + self.config.uncertainty_k_boost * (
            (home_before.uncertainty + away_before.uncertainty)
            / (2.0 * self.config.initial_uncertainty)
            - 1.0
        )
        change = (
            self.k_factor(competition_type)
            * multiplier
            * max(0.5, uncertainty_factor)
            * (actual_home - expected_home)
        )

        home_after = self._updated_state(home_before, change, actual_home - expected_home, date)
        away_after = self._updated_state(away_before, -change, expected_home - actual_home, date)
        self.states[home_team] = home_after
        self.states[away_team] = away_after
        return home_after.rating, away_after.rating

    def process(self, matches: pd.DataFrame) -> pd.DataFrame:
        history: list[dict[str, object]] = []
        for match in matches.itertuples(index=False):
            date = pd.Timestamp(match.date)
            home_state = self.state_for(match.home_team, date)
            away_state = self.state_for(match.away_team, date)
            home_effective = self.effective_rating(
                match.home_team,
                bool(match.neutral_venue),
                True,
                date,
            )
            away_effective = self.effective_rating(
                match.away_team,
                True,
                False,
                date,
            )
            expected_home = self.expected_home_score(
                match.home_team,
                match.away_team,
                bool(match.neutral_venue),
                date,
            )
            home_after, away_after = self.update(
                home_team=match.home_team,
                away_team=match.away_team,
                home_goals=int(match.home_goals),
                away_goals=int(match.away_goals),
                competition_type=match.competition_type,
                neutral_venue=bool(match.neutral_venue),
                date=date,
            )
            history.append(
                {
                    "date": date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "home_goals": int(match.home_goals),
                    "away_goals": int(match.away_goals),
                    "competition_type": match.competition_type,
                    "neutral_venue": bool(match.neutral_venue),
                    "home_rating_before": home_state.rating,
                    "away_rating_before": away_state.rating,
                    "home_effective_rating_before": home_effective,
                    "away_effective_rating_before": away_effective,
                    "home_uncertainty_before": home_state.uncertainty,
                    "away_uncertainty_before": away_state.uncertainty,
                    "home_recent_form_before": home_state.recent_form,
                    "away_recent_form_before": away_state.recent_form,
                    "expected_home_result": expected_home,
                    "home_rating_after": home_after,
                    "away_rating_after": away_after,
                }
            )
        return pd.DataFrame(history)

    def rankings(self) -> list[tuple[str, float]]:
        return sorted(
            ((team, state.rating) for team, state in self.states.items()),
            key=lambda item: item[1],
            reverse=True,
        )

    def ratings(self) -> dict[str, float]:
        return {team: state.rating for team, state in self.states.items()}

    def uncertainties(self) -> dict[str, float]:
        return {team: state.uncertainty for team, state in self.states.items()}

    def recent_forms(self) -> dict[str, float]:
        return {team: state.recent_form for team, state in self.states.items()}

    def _aged_state(self, state: TeamRatingState, date: pd.Timestamp) -> TeamRatingState:
        if state.last_date is None or date <= state.last_date:
            return TeamRatingState(
                rating=state.rating,
                uncertainty=state.uncertainty,
                recent_form=state.recent_form,
                last_date=date,
            )
        elapsed_days = float((date - state.last_date).days)
        uncertainty = min(
            self.config.max_uncertainty,
            state.uncertainty
            + self.config.uncertainty_growth_per_year * elapsed_days / 365.25,
        )
        decay = 0.5 ** (elapsed_days / self.config.recent_half_life_days)
        return TeamRatingState(
            rating=state.rating,
            uncertainty=uncertainty,
            recent_form=state.recent_form * decay,
            last_date=date,
        )

    def _updated_state(
        self,
        state: TeamRatingState,
        rating_change: float,
        surprise: float,
        date: pd.Timestamp | None,
    ) -> TeamRatingState:
        uncertainty = max(
            self.config.min_uncertainty,
            state.uncertainty * self.config.uncertainty_decay_per_match,
        )
        recent_form = 0.85 * state.recent_form + 0.15 * surprise
        return TeamRatingState(
            rating=state.rating + rating_change,
            uncertainty=uncertainty,
            recent_form=recent_form,
            last_date=date,
        )


def _actual_home_result(home_goals: int, away_goals: int) -> float:
    if home_goals > away_goals:
        return 1.0
    if home_goals == away_goals:
        return 0.5
    return 0.0
