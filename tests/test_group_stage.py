from __future__ import annotations

from worldcup_predictor.simulation.group_stage import GroupStanding, rank_group, rank_third_placed


def test_rank_group_uses_points_goal_difference_goals_for() -> None:
    standings = {
        "A": GroupStanding(group="X", team="A", wins=1, goals_for=3, goals_against=1),
        "B": GroupStanding(group="X", team="B", wins=1, goals_for=2, goals_against=0),
        "C": GroupStanding(group="X", team="C", draws=2, goals_for=4, goals_against=4),
        "D": GroupStanding(group="X", team="D", losses=2, goals_for=0, goals_against=4),
    }

    ranked = rank_group(standings)

    assert [standing.team for standing in ranked] == ["A", "B", "C", "D"]


def test_rank_third_placed() -> None:
    ranked = rank_third_placed(
        [
            GroupStanding(group="A", team="A3", wins=1, goals_for=1, goals_against=0),
            GroupStanding(group="B", team="B3", draws=2, goals_for=3, goals_against=3),
            GroupStanding(group="C", team="C3", wins=1, goals_for=2, goals_against=4),
        ]
    )

    assert [standing.group for standing in ranked] == ["A", "C", "B"]

