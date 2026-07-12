from __future__ import annotations

import numpy as np
import pytest

from worldcup_predictor.simulation.batch_tournament import BatchTournamentSimulator
from worldcup_predictor.simulation.tournament import (
    THIRD_PLACE_SLOT_ALLOWED,
    TournamentConfig,
    TournamentSimulator,
    resolve_third_place_slots,
)


class DummyPrediction:
    def __init__(self) -> None:
        self.score_matrix = np.zeros((3, 3), dtype=float).tolist()
        self.score_matrix[1][0] = 1.0


class DummyPredictor:
    def __init__(self, teams: list[str]) -> None:
        self.ratings = {team: 1500.0 for team in teams}

    def predict(self, home_team: str, away_team: str, neutral_venue: bool = True):
        return DummyPrediction()


def test_resolve_third_place_slots_obeys_allowed_groups() -> None:
    assignments = resolve_third_place_slots(list("ACDEFGHI"))

    assert set(assignments) == set(THIRD_PLACE_SLOT_ALLOWED)
    assert set(assignments.values()) == set("ACDEFGHI")
    for slot, group in assignments.items():
        assert group in THIRD_PLACE_SLOT_ALLOWED[slot]


def test_official_annexe_c_mapping_is_loaded() -> None:
    config = TournamentConfig.from_csv(
        "data/worldcup/groups_2026.csv",
        "data/worldcup/fixtures_2026.csv",
    )
    assignments = resolve_third_place_slots(list("EFGHIJKL"), config.third_place_mapping)

    # FIFA Regulations Annexe C, option 1: 1A/1B/1D/1E/1G/1I/1K/1L.
    assert assignments == {
        "T_M79": "E",
        "T_M85": "J",
        "T_M81": "I",
        "T_M74": "F",
        "T_M82": "H",
        "T_M77": "G",
        "T_M87": "L",
        "T_M80": "K",
    }


def test_tournament_simulation_probabilities_close_over_single_run() -> None:
    config = TournamentConfig.from_csv(
        "data/worldcup/groups_2026.csv",
        "data/worldcup/fixtures_2026.csv",
    )
    simulator = TournamentSimulator(
        predictor=DummyPredictor(config.groups["team"].tolist()),
        config=config,
        random_seed=1,
    )

    result = simulator.run(simulations=3)

    assert len(result) == 48
    assert result["champion_prob"].sum() == 1.0
    assert result["group_qualify_prob"].sum() == 32.0
    assert set(result.columns) >= {"team", "round_of_32_prob", "champion_prob"}


def test_simulator_rejects_predictor_missing_team_ratings() -> None:
    config = TournamentConfig.from_csv(
        "data/worldcup/groups_2026.csv",
        "data/worldcup/fixtures_2026.csv",
    )
    teams = config.groups["team"].tolist()
    predictor = DummyPredictor(teams[:-1])

    with pytest.raises(ValueError, match="no rating"):
        TournamentSimulator(predictor=predictor, config=config)


def test_stage_probabilities_use_reach_semantics_and_decrease() -> None:
    config = TournamentConfig.from_csv(
        "data/worldcup/groups_2026.csv",
        "data/worldcup/fixtures_2026.csv",
    )
    simulator = TournamentSimulator(
        predictor=DummyPredictor(config.groups["team"].tolist()),
        config=config,
        random_seed=7,
    )

    result = simulator.run(simulations=5)

    # Exactly 32 teams reach the round of 32, 16 the round of 16, and so on,
    # in every simulation, so the per-stage sums are exact.
    expected_totals = {
        "group_qualify_prob": 32.0,
        "round_of_32_prob": 32.0,
        "round_of_16_prob": 16.0,
        "quarter_final_prob": 8.0,
        "semi_final_prob": 4.0,
        "final_prob": 2.0,
        "champion_prob": 1.0,
    }
    for column, total in expected_totals.items():
        assert result[column].sum() == pytest.approx(total), column

    ordered_columns = [
        "group_qualify_prob",
        "round_of_32_prob",
        "round_of_16_prob",
        "quarter_final_prob",
        "semi_final_prob",
        "final_prob",
        "champion_prob",
    ]
    for _, row in result.iterrows():
        values = [row[column] for column in ordered_columns]
        assert all(
            earlier >= later for earlier, later in zip(values, values[1:])
        ), row["team"]


def test_batch_tournament_simulation_matches_deterministic_simulator() -> None:
    config = TournamentConfig.from_csv(
        "data/worldcup/groups_2026.csv",
        "data/worldcup/fixtures_2026.csv",
    )
    predictor = DummyPredictor(config.groups["team"].tolist())
    baseline = TournamentSimulator(
        predictor=predictor,
        config=config,
        random_seed=1,
    ).run(simulations=3)
    batch = BatchTournamentSimulator(
        predictor=predictor,
        config=config,
        random_seed=1,
    ).run(simulations=3)

    baseline = baseline.sort_values("team").reset_index(drop=True)
    batch = batch.sort_values("team").reset_index(drop=True)

    assert batch["champion_prob"].sum() == 1.0
    assert batch["group_qualify_prob"].sum() == 32.0
    for column in (
        "group_qualify_prob",
        "round_of_32_prob",
        "round_of_16_prob",
        "quarter_final_prob",
        "semi_final_prob",
        "final_prob",
        "champion_prob",
    ):
        assert np.allclose(batch[column], baseline[column]), column
