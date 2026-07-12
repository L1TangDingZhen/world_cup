from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from worldcup_predictor.config import EloConfig
from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.ratings.elo import WorldFootballElo


OUTCOME_LABELS = ("home_win", "draw", "away_win")


def outcome_index(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


class EloLogisticBaseline:
    """Win/draw/loss baseline; intentionally does not model score distributions."""

    model_version = "elo_logistic_baseline_v1"

    def __init__(self, elo_config: EloConfig | None = None) -> None:
        self.elo_config = elo_config or EloConfig()
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )
        self.ratings: dict[str, float] = {}

    def _features(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool,
    ) -> np.ndarray:
        home_rating = self.ratings.get(home_team, self.elo_config.initial_rating)
        away_rating = self.ratings.get(away_team, self.elo_config.initial_rating)
        return np.array(
            [
                (home_rating - away_rating) / self.elo_config.rating_scale,
                0.0 if neutral_venue else 1.0,
            ],
            dtype=float,
        )

    def fit(self, matches: pd.DataFrame) -> "EloLogisticBaseline":
        frame = validate_matches(matches)
        elo = WorldFootballElo(config=self.elo_config)
        features = []
        outcomes = []
        for match in frame.itertuples(index=False):
            self.ratings = dict(elo.ratings)
            features.append(self._features(match.home_team, match.away_team, bool(match.neutral_venue)))
            outcomes.append(outcome_index(int(match.home_goals), int(match.away_goals)))
            elo.update(
                match.home_team,
                match.away_team,
                int(match.home_goals),
                int(match.away_goals),
                match.competition_type,
                bool(match.neutral_venue),
            )
        if len(set(outcomes)) < 2:
            raise ValueError("Baseline training requires at least two outcome classes")
        self.classifier.fit(np.asarray(features), np.asarray(outcomes))
        self.ratings = dict(elo.ratings)
        return self

    def predict_proba(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True,
    ) -> np.ndarray:
        raw = self.classifier.predict_proba(
            self._features(home_team, away_team, neutral_venue).reshape(1, -1)
        )[0]
        probabilities = np.zeros(3, dtype=float)
        probabilities[self.classifier.classes_] = raw
        return probabilities

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
            home_team,
            away_team,
            home_goals,
            away_goals,
            competition_type,
            neutral_venue,
        )
        self.ratings = dict(elo.ratings)
