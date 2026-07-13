from __future__ import annotations

import pytest

from worldcup_predictor.simulation.group_stage import (
    GroupMatch,
    GroupStanding,
    rank_group,
    rank_third_placed,
)


def test_rank_group_uses_points_goal_difference_goals_for() -> None:
    standings = {
        "A": GroupStanding(group="X", team="A", wins=1, goals_for=3, goals_against=1),
        "B": GroupStanding(group="X", team="B", wins=1, goals_for=2, goals_against=0),
        "C": GroupStanding(group="X", team="C", draws=2, goals_for=4, goals_against=4),
        "D": GroupStanding(group="X", team="D", losses=2, goals_for=0, goals_against=4),
    }

    ranked = rank_group(standings)

    assert [standing.team for standing in ranked] == ["A", "B", "C", "D"]


def test_ranking_rule_books_disagree_on_head_to_head_priority() -> None:
    # A and B are level on points; A won the head-to-head, but B has the
    # far better overall goal difference.
    standings = {
        "A": GroupStanding(group="X", team="A", wins=2, losses=1, goals_for=4, goals_against=3),
        "B": GroupStanding(group="X", team="B", wins=2, losses=1, goals_for=7, goals_against=2),
        "C": GroupStanding(group="X", team="C", wins=1, losses=2, goals_for=2, goals_against=4),
        "D": GroupStanding(group="X", team="D", losses=3, goals_for=1, goals_against=5),
    }
    matches = [GroupMatch(home_team="A", away_team="B", home_goals=1, away_goals=0)]

    ranked_2026 = rank_group(standings, matches, rules="fifa_2026")
    ranked_pre_2026 = rank_group(standings, matches, rules="fifa_pre_2026")

    # 2026: head-to-head first, so A leads; 2018/2022: overall goal
    # difference first, so B leads.
    assert [standing.team for standing in ranked_2026][:2] == ["A", "B"]
    assert [standing.team for standing in ranked_pre_2026][:2] == ["B", "A"]


def test_rank_group_rejects_unknown_rules() -> None:
    standings = {
        "A": GroupStanding(group="X", team="A"),
        "B": GroupStanding(group="X", team="B"),
    }
    with pytest.raises(ValueError, match="Unknown ranking rules"):
        rank_group(standings, rules="fifa_1994")


def test_rank_third_placed() -> None:
    ranked = rank_third_placed(
        [
            GroupStanding(group="A", team="A3", wins=1, goals_for=1, goals_against=0),
            GroupStanding(group="B", team="B3", draws=2, goals_for=3, goals_against=3),
            GroupStanding(group="C", team="C3", wins=1, goals_for=2, goals_against=4),
        ]
    )

    assert [standing.group for standing in ranked] == ["A", "C", "B"]

