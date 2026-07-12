from __future__ import annotations

import pytest

from worldcup_predictor.ratings.elo import WorldFootballElo, normalize_competition


def test_elo_update_is_zero_sum() -> None:
    elo = WorldFootballElo()
    home_after, away_after = elo.update(
        home_team="A",
        away_team="B",
        home_goals=2,
        away_goals=0,
        competition_type="World Cup",
        neutral_venue=True,
    )

    assert home_after > 1500
    assert away_after < 1500
    assert home_after + away_after == pytest.approx(3000)


def test_goal_difference_multiplier() -> None:
    assert WorldFootballElo.goal_difference_multiplier(1) == 1.0
    assert WorldFootballElo.goal_difference_multiplier(2) == 1.5
    assert WorldFootballElo.goal_difference_multiplier(4) == 1.875


def test_home_advantage_changes_expected_result() -> None:
    elo = WorldFootballElo()
    neutral = elo.expected_home_score("A", "B", neutral_venue=True)
    home_venue = elo.expected_home_score("A", "B", neutral_venue=False)

    assert neutral == pytest.approx(0.5)
    assert home_venue > neutral


def test_normalize_competition_classifies_real_tournament_names() -> None:
    # Names as they appear in martj42/international_results.
    expected = {
        "FIFA World Cup": "world_cup",
        "FIFA World Cup qualification": "world_cup_qualifier",
        "Friendly": "friendly",
        "UEFA Nations League": "nations_league",
        "CONCACAF Nations League": "nations_league",
        "UEFA Euro": "continental_championship",
        "UEFA Euro qualification": "continental_qualifier",
        "Copa América": "continental_championship",
        "African Cup of Nations": "continental_championship",
        "Africa Cup of Nations": "continental_championship",
        "African Cup of Nations qualification": "continental_qualifier",
        "AFC Asian Cup": "continental_championship",
        "AFC Asian Cup qualification": "continental_qualifier",
        "Gold Cup": "continental_championship",
        "CFU Caribbean Cup qualification": "continental_qualifier",
    }
    for name, bucket in expected.items():
        assert normalize_competition(name) == bucket, name


def test_major_continental_cups_use_championship_k_factor() -> None:
    elo = WorldFootballElo()
    championship_k = elo.config.competition_k_factors["continental_championship"]

    assert elo.k_factor("Copa América") == championship_k
    assert elo.k_factor("African Cup of Nations") == championship_k
    assert elo.k_factor("Friendly") < championship_k

