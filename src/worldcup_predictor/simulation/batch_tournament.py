from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

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
from worldcup_predictor.simulation.tournament import (
    TournamentConfig,
    resolve_third_place_slots,
    validate_predictor_covers_teams,
)


class BatchTournamentSimulator:
    """Experimental batch simulator for measuring vectorized tournament paths.

    This class deliberately does not replace ``TournamentSimulator``.  It keeps
    the same tournament rules but batches score sampling and knockout rounds so
    GPU prediction can be tested without changing the stable simulator.
    """

    def __init__(
        self,
        predictor: Any,
        config: TournamentConfig,
        random_seed: int | None = None,
        device: ComputeDevice = "auto",
        knockout_winners: dict[frozenset[str], str] | None = None,
    ) -> None:
        validate_predictor_covers_teams(predictor, config)
        self.predictor = predictor
        self.config = config
        self.rng = np.random.default_rng(random_seed)
        self.device = device
        self.knockout_winners = dict(knockout_winners or {})
        self._prediction_cache: dict[tuple[str, str, bool], np.ndarray] = {}
        self._predict_accepts_device = "device" in inspect.signature(
            predictor.predict
        ).parameters
        predict_many = getattr(predictor, "predict_many", None)
        self._predict_many = predict_many if callable(predict_many) else None
        self._predict_many_accepts_device = (
            self._predict_many is not None
            and "device" in inspect.signature(self._predict_many).parameters
        )

    def run(self, simulations: int = 1000) -> pd.DataFrame:
        if simulations <= 0:
            raise ValueError("simulations must be positive")

        tournament_format = self.config.format
        teams = self.config.groups["team"].tolist()
        counters = {
            team: {stage: 0 for stage in tournament_format.stage_columns}
            for team in teams
        }

        ranked_group_runs = self._simulate_group_stage_batch(simulations)
        selectors_by_run: list[dict[str, str]] = []
        for ranked_groups in ranked_group_runs:
            selectors: dict[str, str] = {}
            thirds: list[GroupStanding] = []
            for group, standings in ranked_groups.items():
                selectors[f"W_{group}"] = standings[0].team
                selectors[f"R_{group}"] = standings[1].team
                thirds.append(standings[2])

            qualified = set(selectors[f"W_{group}"] for group in ranked_groups)
            qualified.update(selectors[f"R_{group}"] for group in ranked_groups)

            if tournament_format.third_place is not None:
                qualified_thirds = rank_third_placed(thirds)[
                    : tournament_format.third_place.qualifier_count
                ]
                third_assignments = resolve_third_place_slots(
                    [standing.group for standing in qualified_thirds],
                    self.config.third_place_mapping,
                )
                third_by_group = {
                    standing.group: standing.team for standing in qualified_thirds
                }
                for slot, group in third_assignments.items():
                    selectors[slot] = third_by_group[group]
                qualified.update(third_by_group.values())

            for team in qualified:
                counters[team]["group_qualify"] += 1
                counters[team][tournament_format.qualification_stage] += 1
            selectors_by_run.append(selectors)

        winners_by_run = [dict[int, str]() for _ in range(simulations)]
        entry_pairs = [
            (selectors[left_selector], selectors[right_selector])
            for selectors in selectors_by_run
            for _, left_selector, right_selector in tournament_format.entry_matches
        ]
        entry_winners = self._sample_knockout_winners(entry_pairs)
        cursor = 0
        for winners in winners_by_run:
            for match_id, _, _ in tournament_format.entry_matches:
                winners[match_id] = entry_winners[cursor]
                counters[winners[match_id]][
                    tournament_format.entry_winners_reach
                ] += 1
                cursor += 1

        for round_index, round_matches in enumerate(tournament_format.bracket_rounds):
            stage = tournament_format.stage_reached_by_bracket_round.get(round_index)
            pairs = [
                (winners[left_match], winners[right_match])
                for winners in winners_by_run
                for _, left_match, right_match in round_matches
            ]
            sampled_winners = self._sample_knockout_winners(pairs)
            cursor = 0
            for winners in winners_by_run:
                for match_id, _, _ in round_matches:
                    winner = sampled_winners[cursor]
                    winners[match_id] = winner
                    if stage is not None:
                        counters[winner][stage] += 1
                    cursor += 1

        for winners in winners_by_run:
            counters[winners[tournament_format.final_match_id]]["champion"] += 1

        rows = []
        for team, values in counters.items():
            row = {"team": team}
            row.update(
                {key + "_prob": value / simulations for key, value in values.items()}
            )
            rows.append(row)
        return (
            pd.DataFrame(rows)
            .sort_values("champion_prob", ascending=False)
            .reset_index(drop=True)
        )

    def _simulate_group_stage_batch(
        self,
        simulations: int,
    ) -> list[dict[str, list[GroupStanding]]]:
        fixtures = list(self.config.fixtures.itertuples(index=False))
        fixture_scores = self._sample_group_fixture_scores(fixtures, simulations)
        ranked_group_runs = []

        for simulation_index in range(simulations):
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
            group_matches: dict[str, list[GroupMatch]] = {
                group: [] for group in standings
            }
            for fixture, (home_scores, away_scores) in zip(
                fixtures,
                fixture_scores,
                strict=True,
            ):
                home_goals = int(home_scores[simulation_index])
                away_goals = int(away_scores[simulation_index])
                home_fair_play = int(getattr(fixture, "home_fair_play", 0) or 0)
                away_fair_play = int(getattr(fixture, "away_fair_play", 0) or 0)
                standings[fixture.group][fixture.home_team].add_match(
                    home_goals,
                    away_goals,
                    home_fair_play,
                )
                standings[fixture.group][fixture.away_team].add_match(
                    away_goals,
                    home_goals,
                    away_fair_play,
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
            ranked_group_runs.append(
                {
                    group: rank_group(
                        group_standings,
                        group_matches[group],
                        rules=self.config.format.ranking_rules,
                    )
                    for group, group_standings in standings.items()
                }
            )
        return ranked_group_runs

    def _sample_group_fixture_scores(
        self,
        fixtures: list[Any],
        simulations: int,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        score_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        needed = [
            (fixture.home_team, fixture.away_team, bool(fixture.neutral_venue))
            for fixture in fixtures
            if pd.isna(fixture.home_goals) or pd.isna(fixture.away_goals)
        ]
        self._score_matrices_for_matches(needed)

        for fixture in fixtures:
            if pd.notna(fixture.home_goals) and pd.notna(fixture.away_goals):
                score_pairs.append(
                    (
                        np.full(simulations, int(fixture.home_goals), dtype=np.int16),
                        np.full(simulations, int(fixture.away_goals), dtype=np.int16),
                    )
                )
                continue
            matrix = self._prediction_cache[
                (fixture.home_team, fixture.away_team, bool(fixture.neutral_venue))
            ]
            score_pairs.append(self._sample_scores(matrix, simulations))
        return score_pairs

    def _sample_knockout_winners(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[str]:
        if not pairs:
            return []
        winners = np.empty(len(pairs), dtype=object)
        unpinned_indices: list[int] = []
        for index, (team_a, team_b) in enumerate(pairs):
            pinned_winner = (
                self.knockout_winners.get(frozenset((team_a, team_b)))
                if self.knockout_winners
                else None
            )
            if pinned_winner is not None:
                winners[index] = pinned_winner
            else:
                unpinned_indices.append(index)
        if not unpinned_indices:
            return winners.tolist()

        keys = [
            (pairs[index][0], pairs[index][1], True) for index in unpinned_indices
        ]
        self._score_matrices_for_matches(keys)

        indices_by_key: dict[tuple[str, str, bool], list[int]] = {}
        for key, index in zip(keys, unpinned_indices, strict=True):
            indices_by_key.setdefault(key, []).append(index)
        for key, indices in indices_by_key.items():
            team_a, team_b, _ = key
            index_array = np.asarray(indices, dtype=np.int64)
            home_goals, away_goals = self._sample_scores(
                self._prediction_cache[key],
                len(indices),
            )
            home_wins = home_goals > away_goals
            away_wins = away_goals > home_goals
            ties = ~(home_wins | away_wins)
            winners[index_array[home_wins]] = team_a
            winners[index_array[away_wins]] = team_b
            if np.any(ties):
                tie_indices = index_array[ties]
                probability_a = self._penalty_probability(team_a, team_b)
                winners[tie_indices] = np.where(
                    self.rng.random(len(tie_indices)) < probability_a,
                    team_a,
                    team_b,
                )
        return winners.tolist()

    def _score_matrices_for_matches(
        self,
        matches: list[tuple[str, str, bool]],
    ) -> None:
        missing = list(dict.fromkeys(match for match in matches if match not in self._prediction_cache))
        if not missing:
            return

        if self._predict_many is not None:
            if self._predict_many_accepts_device:
                predictions = self._predict_many(missing, device=self.device)
            else:
                predictions = self._predict_many(missing)
        else:
            predictions = []
            for home_team, away_team, neutral_venue in missing:
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
                predictions.append(prediction)

        for match, prediction in zip(missing, predictions, strict=True):
            matrix = np.asarray(prediction.score_matrix, dtype=np.float64)
            total = matrix.sum()
            if total <= 0:
                raise RuntimeError("Score matrix has zero probability mass")
            self._prediction_cache[match] = matrix / total

    def _sample_scores(
        self,
        matrix: np.ndarray,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        probabilities = matrix.reshape(-1)
        choices = self.rng.choice(
            probabilities.size,
            size=size,
            p=probabilities / probabilities.sum(),
        )
        return np.unravel_index(choices, matrix.shape)

    def _penalty_probability(self, team_a: str, team_b: str) -> float:
        rating_a = self.predictor.ratings.get(team_a, 1500.0)
        rating_b = self.predictor.ratings.get(team_b, 1500.0)
        return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 800.0))


def load_batch_elo_poisson_simulator(
    model_path: str | Path,
    groups_path: str | Path,
    fixtures_path: str | Path,
    random_seed: int | None = None,
    device: ComputeDevice = "auto",
) -> BatchTournamentSimulator:
    return BatchTournamentSimulator(
        predictor=EloPoissonModel.load(model_path),
        config=TournamentConfig.from_csv(groups_path, fixtures_path),
        random_seed=random_seed,
        device=device,
    )
