from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_predictor.ingestion.matches import validate_matches
from worldcup_predictor.simulation.actual_results import (
    load_actual_knockout_winners,
)
from worldcup_predictor.simulation.batch_tournament import BatchTournamentSimulator
from worldcup_predictor.simulation.tournament import (
    TournamentConfig,
    TournamentSimulator,
)


class HomeWinsPrediction:
    def __init__(self) -> None:
        self.score_matrix = np.zeros((3, 3), dtype=float).tolist()
        self.score_matrix[1][0] = 1.0


class HomeWinsPredictor:
    def __init__(self, teams: list[str]) -> None:
        self.ratings = {team: 1500.0 for team in teams}

    def predict(self, home_team: str, away_team: str, neutral_venue: bool = True):
        return HomeWinsPrediction()


def _config() -> TournamentConfig:
    return TournamentConfig.from_csv(
        "data/worldcup/groups_2026.csv",
        "data/worldcup/fixtures_2026.csv",
    )


def _knockout_matches(include_draw: bool = True) -> pd.DataFrame:
    rows = [
        {
            # Decisive knockout result: winner comes straight from the score.
            "date": "2026-06-29",
            "home_team": "Spain",
            "away_team": "England",
            "home_goals": 2,
            "away_goals": 1,
            "competition_type": "FIFA World Cup",
            "neutral_venue": True,
        },
        {
            # Friendly after the group stage must be ignored.
            "date": "2026-06-29",
            "home_team": "France",
            "away_team": "Argentina",
            "home_goals": 4,
            "away_goals": 0,
            "competition_type": "Friendly",
            "neutral_venue": True,
        },
        {
            # World Cup match involving a non-tournament team must be ignored.
            "date": "2026-06-29",
            "home_team": "Wales",
            "away_team": "Brazil",
            "home_goals": 1,
            "away_goals": 0,
            "competition_type": "FIFA World Cup",
            "neutral_venue": True,
        },
    ]
    if include_draw:
        rows.append(
            {
                # Drawn after full time: decided on penalties.
                "date": "2026-06-30",
                "home_team": "France",
                "away_team": "Brazil",
                "home_goals": 1,
                "away_goals": 1,
                "competition_type": "FIFA World Cup",
                "neutral_venue": True,
            }
        )
    return validate_matches(pd.DataFrame(rows))


def test_loader_maps_decisive_results_and_shootout_winners() -> None:
    shootouts = pd.DataFrame(
        [
            {
                "date": "2026-06-30",
                "home_team": "France",
                "away_team": "Brazil",
                "winner": "Brazil",
            }
        ]
    )

    winners = load_actual_knockout_winners(
        _knockout_matches(), _config(), shootouts=shootouts
    )

    assert winners == {
        frozenset({"Spain", "England"}): "Spain",
        frozenset({"France", "Brazil"}): "Brazil",
    }


def test_loader_raises_for_draw_without_shootout_winner() -> None:
    with pytest.raises(ValueError, match="no shootout winner"):
        load_actual_knockout_winners(_knockout_matches(), _config(), shootouts=None)


def test_simulator_pins_actual_knockout_winner() -> None:
    config = _config()
    predictor = HomeWinsPredictor(config.groups["team"].tolist())

    baseline = TournamentSimulator(predictor=predictor, config=config, random_seed=3)
    outcome = baseline.simulate_once()
    finalists = sorted(outcome["stage_advancers"]["final"])
    champion = outcome["champion"]
    runner_up = next(team for team in finalists if team != champion)

    pinned = TournamentSimulator(
        predictor=predictor,
        config=config,
        random_seed=3,
        knockout_winners={frozenset(finalists): runner_up},
    )

    assert pinned.simulate_once()["champion"] == runner_up


def test_batch_simulator_pins_actual_knockout_winner() -> None:
    config = _config()
    predictor = HomeWinsPredictor(config.groups["team"].tolist())
    outcome = TournamentSimulator(
        predictor=predictor, config=config, random_seed=3
    ).simulate_once()
    finalists = sorted(outcome["stage_advancers"]["final"])
    runner_up = next(team for team in finalists if team != outcome["champion"])

    result = BatchTournamentSimulator(
        predictor=predictor,
        config=config,
        random_seed=3,
        knockout_winners={frozenset(finalists): runner_up},
    ).run(simulations=2)

    champion_row = result.loc[result["team"] == runner_up]
    assert champion_row["champion_prob"].iloc[0] == 1.0


def test_run_warns_when_actual_results_do_not_match_bracket() -> None:
    config = _config()
    predictor = HomeWinsPredictor(config.groups["team"].tolist())
    simulator = TournamentSimulator(
        predictor=predictor,
        config=config,
        random_seed=3,
        knockout_winners={frozenset({"Foo", "Bar"}): "Foo"},
    )

    with pytest.warns(UserWarning, match="never matched a simulated pairing"):
        simulator.run(simulations=1)
