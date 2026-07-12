from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import inspect
from typing import Protocol

import numpy as np
import pandas as pd

from worldcup_predictor.compute import ComputeDevice
from worldcup_predictor.models.elo_poisson import EloPoissonModel
from worldcup_predictor.simulation.group_stage import (
    GroupMatch,
    GroupStanding,
    rank_group,
    rank_third_placed,
)


class MatchPredictor(Protocol):
    ratings: dict[str, float]

    def predict(self, home_team: str, away_team: str, neutral_venue: bool = True):
        ...


ROUND_OF_32_MATCHES = [
    (73, "R_A", "R_B"),
    (74, "W_E", "T_M74"),
    (75, "W_F", "R_C"),
    (76, "W_C", "R_F"),
    (77, "W_I", "T_M77"),
    (78, "R_E", "R_I"),
    (79, "W_A", "T_M79"),
    (80, "W_L", "T_M80"),
    (81, "W_D", "T_M81"),
    (82, "W_G", "T_M82"),
    (83, "R_K", "R_L"),
    (84, "W_H", "R_J"),
    (85, "W_B", "T_M85"),
    (86, "W_J", "R_H"),
    (87, "W_K", "T_M87"),
    (88, "R_D", "R_G"),
]

THIRD_PLACE_SLOT_ALLOWED = {
    "T_M74": set("ABCDF"),
    "T_M77": set("CDFGH"),
    "T_M79": set("CEFHI"),
    "T_M80": set("EHIJK"),
    "T_M81": set("BEFIJ"),
    "T_M82": set("AEHIJ"),
    "T_M85": set("EFGIJ"),
    "T_M87": set("DEIJL"),
}

BRACKET_ROUNDS = [
    [(89, 74, 77), (90, 73, 75), (91, 76, 78), (92, 79, 80),
     (93, 83, 84), (94, 81, 82), (95, 86, 88), (96, 85, 87)],
    [(97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)],
    [(101, 97, 98), (102, 99, 100)],
    [(103, 101, 102)],
]

# Stage probabilities use "reached this stage" semantics: winning a match at
# one stage means reaching the next one.  Winners of the bracket rounds below
# (round of 16, quarter-finals, semi-finals) therefore count toward the next
# stage; winners of the round-of-32 matches count toward "round_of_16", and
# the winner of the final only counts toward "champion".
STAGE_REACHED_BY_BRACKET_ROUND = {
    0: "quarter_final",
    1: "semi_final",
    2: "final",
}


@dataclass(frozen=True)
class TournamentConfig:
    groups: pd.DataFrame
    fixtures: pd.DataFrame
    third_place_mapping: dict[str, dict[str, str]]

    @classmethod
    def from_csv(
        cls,
        groups_path: str | Path,
        fixtures_path: str | Path,
        mapping_path: str | Path = "data/worldcup/third_place_mapping_2026.csv",
    ) -> "TournamentConfig":
        groups = pd.read_csv(groups_path)
        fixtures = pd.read_csv(fixtures_path)
        required_groups = {"group", "team"}
        required_fixtures = {
            "group",
            "date",
            "home_team",
            "away_team",
            "neutral_venue",
        }
        missing_groups = required_groups - set(groups.columns)
        missing_fixtures = required_fixtures - set(fixtures.columns)
        if missing_groups:
            raise ValueError(f"Missing group columns: {sorted(missing_groups)}")
        if missing_fixtures:
            raise ValueError(f"Missing fixture columns: {sorted(missing_fixtures)}")
        groups = groups.copy()
        fixtures = fixtures.copy()
        groups["group"] = groups["group"].astype(str).str.strip()
        groups["team"] = groups["team"].astype(str).str.strip()
        fixtures["group"] = fixtures["group"].astype(str).str.strip()
        fixtures["home_team"] = fixtures["home_team"].astype(str).str.strip()
        fixtures["away_team"] = fixtures["away_team"].astype(str).str.strip()
        fixtures["date"] = pd.to_datetime(fixtures["date"])
        if "home_goals" not in fixtures.columns:
            fixtures["home_goals"] = np.nan
        if "away_goals" not in fixtures.columns:
            fixtures["away_goals"] = np.nan
        fixtures["neutral_venue"] = fixtures["neutral_venue"].map(_parse_bool)
        if "fifa_ranking" not in groups.columns:
            groups["fifa_ranking"] = 999
        if "home_fair_play" not in fixtures.columns:
            fixtures["home_fair_play"] = 0
        if "away_fair_play" not in fixtures.columns:
            fixtures["away_fair_play"] = 0

        for group, teams in groups.groupby("group")["team"]:
            if len(teams) != 4:
                raise ValueError(f"Group {group} must contain exactly 4 teams")
        mapping_frame = pd.read_csv(mapping_path)
        expected_mapping_columns = {"qualifying_groups", *THIRD_PLACE_SLOT_ALLOWED}
        missing_mapping = expected_mapping_columns - set(mapping_frame.columns)
        if missing_mapping:
            raise ValueError(f"Missing third-place mapping columns: {sorted(missing_mapping)}")
        mapping = {
            str(row.qualifying_groups): {
                slot: str(getattr(row, slot))
                for slot in THIRD_PLACE_SLOT_ALLOWED
            }
            for row in mapping_frame.itertuples(index=False)
        }
        if len(mapping) != 495:
            raise ValueError(f"Expected 495 official third-place mappings, found {len(mapping)}")
        return cls(
            groups=groups.sort_values(["group", "team"]).reset_index(drop=True),
            fixtures=fixtures.sort_values("date", kind="stable").reset_index(drop=True),
            third_place_mapping=mapping,
        )


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def validate_predictor_covers_teams(predictor: object, config: TournamentConfig) -> None:
    """Fail fast when tournament teams are missing from the predictor.

    Without this check a misspelled team would silently be treated as an
    average side at the initial rating.
    """
    ratings = getattr(predictor, "ratings", None)
    if not isinstance(ratings, Mapping):
        return
    missing = sorted(set(config.groups["team"]) - set(ratings))
    if missing:
        raise ValueError(
            f"Predictor has no rating for {len(missing)} tournament team(s): "
            f"{', '.join(missing)}"
        )


class TournamentSimulator:
    def __init__(
        self,
        predictor: MatchPredictor,
        config: TournamentConfig,
        random_seed: int | None = None,
        device: ComputeDevice = "auto",
        knockout_winners: Mapping[frozenset[str], str] | None = None,
    ) -> None:
        validate_predictor_covers_teams(predictor, config)
        self.predictor = predictor
        self.config = config
        self.rng = np.random.default_rng(random_seed)
        self.device = device
        # Real knockout results keyed by the (unordered) pairing; pairings
        # that already happened are pinned instead of sampled.
        self.knockout_winners = dict(knockout_winners or {})
        self._used_knockout_pairs: set[frozenset[str]] = set()
        self._predict_accepts_device = "device" in inspect.signature(
            predictor.predict
        ).parameters
        self._prediction_cache: dict[tuple[str, str, bool], object] = {}

    def run(self, simulations: int = 1000) -> pd.DataFrame:
        if simulations <= 0:
            raise ValueError("simulations must be positive")
        teams = self.config.groups["team"].tolist()
        counters = {
            team: {
                "group_qualify": 0,
                "round_of_32": 0,
                "round_of_16": 0,
                "quarter_final": 0,
                "semi_final": 0,
                "final": 0,
                "champion": 0,
            }
            for team in teams
        }
        for _ in range(simulations):
            result = self.simulate_once()
            for team in result["qualified"]:
                counters[team]["group_qualify"] += 1
                counters[team]["round_of_32"] += 1
            for stage, stage_teams in result["stage_advancers"].items():
                for team in stage_teams:
                    counters[team][stage] += 1
            counters[result["champion"]]["champion"] += 1

        if self.knockout_winners:
            unused = set(self.knockout_winners) - self._used_knockout_pairs
            if unused:
                pairings = ", ".join(
                    " vs ".join(sorted(pair))
                    for pair in sorted(unused, key=lambda item: sorted(item))
                )
                warnings.warn(
                    f"{len(unused)} actual knockout result(s) never matched a "
                    f"simulated pairing: {pairings}. The simulated bracket may "
                    "not line up with reality; check that all group results "
                    "are filled in the fixtures file.",
                    stacklevel=2,
                )

        rows = []
        for team, values in counters.items():
            row = {"team": team}
            row.update({key + "_prob": value / simulations for key, value in values.items()})
            rows.append(row)
        return pd.DataFrame(rows).sort_values("champion_prob", ascending=False).reset_index(drop=True)

    def simulate_once(self) -> dict[str, object]:
        ranked_groups = self._simulate_group_stage()
        selectors: dict[str, str] = {}
        thirds: list[GroupStanding] = []
        for group, standings in ranked_groups.items():
            selectors[f"W_{group}"] = standings[0].team
            selectors[f"R_{group}"] = standings[1].team
            thirds.append(standings[2])

        qualified_thirds = rank_third_placed(thirds)[:8]
        third_assignments = resolve_third_place_slots(
            [standing.group for standing in qualified_thirds],
            self.config.third_place_mapping,
        )
        third_by_group = {standing.group: standing.team for standing in qualified_thirds}
        for slot, group in third_assignments.items():
            selectors[slot] = third_by_group[group]

        winners: dict[int, str] = {}
        qualified = set(selectors[f"W_{group}"] for group in ranked_groups)
        qualified.update(selectors[f"R_{group}"] for group in ranked_groups)
        qualified.update(third_by_group.values())

        stage_advancers: dict[str, set[str]] = {
            "round_of_16": set(),
            **{stage: set() for stage in STAGE_REACHED_BY_BRACKET_ROUND.values()},
        }
        for match_id, left_selector, right_selector in ROUND_OF_32_MATCHES:
            winners[match_id] = self._simulate_knockout_match(
                selectors[left_selector],
                selectors[right_selector],
            )
            stage_advancers["round_of_16"].add(winners[match_id])

        for round_index, round_matches in enumerate(BRACKET_ROUNDS):
            stage = STAGE_REACHED_BY_BRACKET_ROUND.get(round_index)
            for match_id, left_match, right_match in round_matches:
                winners[match_id] = self._simulate_knockout_match(
                    winners[left_match],
                    winners[right_match],
                )
                if stage is not None:
                    stage_advancers[stage].add(winners[match_id])

        return {
            "ranked_groups": ranked_groups,
            "qualified": qualified,
            "stage_advancers": stage_advancers,
            "champion": winners[103],
        }

    def _simulate_group_stage(self) -> dict[str, list[GroupStanding]]:
        standings = {
            group: {
                row.team: GroupStanding(
                    group=group,
                    team=row.team,
                    fifa_ranking=int(row.fifa_ranking),
                )
                for row in teams.itertuples(index=False)
            }
            for group, teams in self.config.groups.groupby("group")
        }
        group_matches: dict[str, list[GroupMatch]] = {group: [] for group in standings}
        for fixture in self.config.fixtures.itertuples(index=False):
            home_goals, away_goals = self._score_fixture(fixture)
            home_fair_play = int(getattr(fixture, "home_fair_play", 0) or 0)
            away_fair_play = int(getattr(fixture, "away_fair_play", 0) or 0)
            standings[fixture.group][fixture.home_team].add_match(
                home_goals, away_goals, home_fair_play
            )
            standings[fixture.group][fixture.away_team].add_match(
                away_goals, home_goals, away_fair_play
            )
            group_matches[fixture.group].append(
                GroupMatch(
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    home_fair_play=home_fair_play,
                    away_fair_play=away_fair_play,
                )
            )
        return {
            group: rank_group(group_standings, group_matches[group])
            for group, group_standings in standings.items()
        }

    def _score_fixture(self, fixture) -> tuple[int, int]:
        if pd.notna(fixture.home_goals) and pd.notna(fixture.away_goals):
            return int(fixture.home_goals), int(fixture.away_goals)
        prediction = self._predict(
            fixture.home_team,
            fixture.away_team,
            neutral_venue=bool(fixture.neutral_venue),
        )
        matrix = np.asarray(prediction.score_matrix, dtype=float)
        choice = int(self.rng.choice(matrix.size, p=matrix.reshape(-1)))
        return tuple(int(value) for value in np.unravel_index(choice, matrix.shape))

    def _simulate_knockout_match(self, team_a: str, team_b: str) -> str:
        if self.knockout_winners:
            pair = frozenset((team_a, team_b))
            pinned_winner = self.knockout_winners.get(pair)
            if pinned_winner is not None:
                self._used_knockout_pairs.add(pair)
                return pinned_winner
        prediction = self._predict(team_a, team_b, neutral_venue=True)
        matrix = np.asarray(prediction.score_matrix, dtype=float)
        choice = int(self.rng.choice(matrix.size, p=matrix.reshape(-1)))
        goals_a, goals_b = np.unravel_index(choice, matrix.shape)
        if goals_a > goals_b:
            return team_a
        if goals_b > goals_a:
            return team_b
        return self._penalty_winner(team_a, team_b)

    def _penalty_winner(self, team_a: str, team_b: str) -> str:
        rating_a = self.predictor.ratings.get(team_a, 1500.0)
        rating_b = self.predictor.ratings.get(team_b, 1500.0)
        probability_a = 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 800.0))
        return team_a if self.rng.random() < probability_a else team_b

    def _predict(self, home_team: str, away_team: str, neutral_venue: bool):
        cache_key = (home_team, away_team, neutral_venue)
        cached = self._prediction_cache.get(cache_key)
        if cached is not None:
            return cached
        if self._predict_accepts_device:
            prediction = self.predictor.predict(
                home_team,
                away_team,
                neutral_venue=neutral_venue,
                device=self.device,
            )
        else:
            prediction = self.predictor.predict(
                home_team,
                away_team,
                neutral_venue=neutral_venue,
            )
        self._prediction_cache[cache_key] = prediction
        return prediction


def resolve_third_place_slots(
    qualified_groups: list[str],
    official_mapping: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    groups = sorted({group.strip() for group in qualified_groups})
    if len(groups) != 8:
        raise ValueError("Exactly eight third-placed groups must qualify")

    key = "".join(groups)
    if official_mapping is not None:
        if key not in official_mapping:
            raise ValueError(f"No FIFA Annexe C mapping for groups {key}")
        return dict(official_mapping[key])

    slots = sorted(THIRD_PLACE_SLOT_ALLOWED)
    assignment: dict[str, str] = {}
    used: set[str] = set()

    def backtrack() -> bool:
        if len(assignment) == len(slots):
            return True
        remaining_slots = [slot for slot in slots if slot not in assignment]
        slot = min(
            remaining_slots,
            key=lambda item: len(THIRD_PLACE_SLOT_ALLOWED[item] & set(groups) - used),
        )
        candidates = sorted((THIRD_PLACE_SLOT_ALLOWED[slot] & set(groups)) - used)
        for group in candidates:
            assignment[slot] = group
            used.add(group)
            if backtrack():
                return True
            used.remove(group)
            del assignment[slot]
        return False

    if not backtrack():
        raise ValueError(f"No valid third-place slot assignment for groups {groups}")
    return assignment


def load_elo_poisson_simulator(
    model_path: str | Path,
    groups_path: str | Path,
    fixtures_path: str | Path,
    random_seed: int | None = None,
    device: ComputeDevice = "auto",
    knockout_winners: Mapping[frozenset[str], str] | None = None,
) -> TournamentSimulator:
    return TournamentSimulator(
        predictor=EloPoissonModel.load(model_path),
        config=TournamentConfig.from_csv(groups_path, fixtures_path),
        random_seed=random_seed,
        device=device,
        knockout_winners=knockout_winners,
    )
