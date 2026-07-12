from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class GroupStanding:
    group: str
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    fair_play_score: int = 0
    fifa_ranking: int = 999

    @property
    def points(self) -> int:
        return self.wins * 3 + self.draws

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against

    def add_match(
        self,
        goals_for: int,
        goals_against: int,
        fair_play_delta: int = 0,
    ) -> None:
        self.played += 1
        self.goals_for += goals_for
        self.goals_against += goals_against
        self.fair_play_score += fair_play_delta
        if goals_for > goals_against:
            self.wins += 1
        elif goals_for == goals_against:
            self.draws += 1
        else:
            self.losses += 1


@dataclass(frozen=True)
class GroupMatch:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    home_fair_play: int = 0
    away_fair_play: int = 0


def _mini_table(
    teams: set[str],
    matches: Iterable[GroupMatch],
    standings: dict[str, GroupStanding],
) -> dict[str, GroupStanding]:
    mini = {
        team: GroupStanding(
            group=standings[team].group,
            team=team,
            fair_play_score=standings[team].fair_play_score,
            fifa_ranking=standings[team].fifa_ranking,
        )
        for team in teams
    }
    for match in matches:
        if match.home_team in teams and match.away_team in teams:
            mini[match.home_team].add_match(match.home_goals, match.away_goals)
            mini[match.away_team].add_match(match.away_goals, match.home_goals)
    return mini


def _partition_by_key(teams: list[str], key) -> list[list[str]]:
    groups: dict[tuple[int, ...], list[str]] = {}
    for team in teams:
        groups.setdefault(key(team), []).append(team)
    return [groups[value] for value in sorted(groups, reverse=True)]


def _rank_tied_teams(
    teams: list[str],
    standings: dict[str, GroupStanding],
    matches: list[GroupMatch],
) -> list[str]:
    if len(teams) <= 1:
        return teams
    mini = _mini_table(set(teams), matches, standings)
    mini_key = lambda team: (
        mini[team].points,
        mini[team].goal_difference,
        mini[team].goals_for,
    )
    partitions = _partition_by_key(teams, mini_key)
    if len(partitions) > 1:
        return [team for partition in partitions for team in _rank_tied_teams(partition, standings, matches)]

    # FIFA Article 13 step 2/3 after head-to-head remains tied.
    return sorted(
        teams,
        key=lambda team: (
            -standings[team].goal_difference,
            -standings[team].goals_for,
            -standings[team].fair_play_score,
            standings[team].fifa_ranking,
            team,
        ),
    )


def rank_group(
    standings: dict[str, GroupStanding],
    matches: list[GroupMatch] | None = None,
) -> list[GroupStanding]:
    """Rank a group under FIFA World Cup 2026 Article 13."""
    matches = matches or []
    points_partitions = _partition_by_key(
        list(standings),
        lambda team: (standings[team].points,),
    )
    ordered_teams = [
        team
        for partition in points_partitions
        for team in _rank_tied_teams(partition, standings, matches)
    ]
    return [standings[team] for team in ordered_teams]


def rank_third_placed(third_placed: list[GroupStanding]) -> list[GroupStanding]:
    """Rank third-placed teams under FIFA Article 13, criteria a-f."""
    return sorted(
        third_placed,
        key=lambda standing: (
            -standing.points,
            -standing.goal_difference,
            -standing.goals_for,
            -standing.fair_play_score,
            standing.fifa_ranking,
            standing.team,
        ),
    )
