from __future__ import annotations

import pytest

from worldcup_predictor.simulation.formats import (
    FORMATS,
    TournamentFormat,
    WC32,
    WC48_2026,
    get_format,
)


def _assert_bracket_wiring(tournament_format: TournamentFormat) -> None:
    """Every match winner must feed exactly one match of the next round."""
    previous_ids = [match_id for match_id, _, _ in tournament_format.entry_matches]
    assert len(previous_ids) == len(set(previous_ids))
    for round_matches in tournament_format.bracket_rounds:
        inputs = [left for _, left, _ in round_matches] + [
            right for _, _, right in round_matches
        ]
        assert sorted(inputs) == sorted(previous_ids)
        previous_ids = [match_id for match_id, _, _ in round_matches]
    assert len(previous_ids) == 1
    assert previous_ids[0] == tournament_format.final_match_id


def test_bracket_wiring_is_consistent_for_all_formats() -> None:
    for tournament_format in FORMATS.values():
        _assert_bracket_wiring(tournament_format)


def test_stage_columns_and_structural_counts() -> None:
    assert WC48_2026.stage_columns == (
        "group_qualify",
        "round_of_32",
        "round_of_16",
        "quarter_final",
        "semi_final",
        "final",
        "champion",
    )
    assert WC48_2026.teams_reaching == {
        "group_qualify": 32,
        "round_of_32": 32,
        "round_of_16": 16,
        "quarter_final": 8,
        "semi_final": 4,
        "final": 2,
        "champion": 1,
    }
    assert WC32.stage_columns == (
        "group_qualify",
        "round_of_16",
        "quarter_final",
        "semi_final",
        "final",
        "champion",
    )
    assert WC32.teams_reaching == {
        "group_qualify": 16,
        "round_of_16": 16,
        "quarter_final": 8,
        "semi_final": 4,
        "final": 2,
        "champion": 1,
    }


def test_entry_selectors_cover_all_groups() -> None:
    for tournament_format in FORMATS.values():
        groups = {chr(ord("A") + index) for index in range(tournament_format.group_count)}
        winners = {f"W_{group}" for group in groups}
        runners = {f"R_{group}" for group in groups}
        third_slots = (
            set(tournament_format.third_place.slot_allowed_groups)
            if tournament_format.third_place
            else set()
        )
        selectors = [
            selector
            for _, left, right in tournament_format.entry_matches
            for selector in (left, right)
        ]
        assert len(selectors) == len(set(selectors))
        assert set(selectors) <= winners | runners | third_slots
        # Every group winner and runner-up enters the knockout bracket.
        assert winners | runners <= set(selectors)


def test_get_format_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown tournament format"):
        get_format("wc16_1978")
