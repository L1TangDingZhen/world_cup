"""Tournament format definitions.

A TournamentFormat describes everything competition-specific about a World
Cup edition: how many groups there are, which ranking rule book applies,
how knockout entry matches are seeded from group positions (including the
best-third-placed mechanism when the edition has one), and how the bracket
rounds chain together. The simulators consume only this object, so new
editions are added here without touching simulation logic.

Selectors used in entry matches: ``W_A``/``R_A`` are the winner and
runner-up of group A; ``T_M74`` style selectors are third-placed slots
resolved through the edition's third-place mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

# --- FIFA World Cup 2026 (48 teams, 12 groups) ------------------------------

ROUND_OF_32_MATCHES = (
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
)

THIRD_PLACE_SLOT_ALLOWED = {
    "T_M74": frozenset("ABCDF"),
    "T_M77": frozenset("CDFGH"),
    "T_M79": frozenset("CEFHI"),
    "T_M80": frozenset("EHIJK"),
    "T_M81": frozenset("BEFIJ"),
    "T_M82": frozenset("AEHIJ"),
    "T_M85": frozenset("EFGIJ"),
    "T_M87": frozenset("DEIJL"),
}

BRACKET_ROUNDS_2026 = (
    (
        (89, 74, 77),
        (90, 73, 75),
        (91, 76, 78),
        (92, 79, 80),
        (93, 83, 84),
        (94, 81, 82),
        (95, 86, 88),
        (96, 85, 87),
    ),
    ((97, 89, 90), (98, 93, 94), (99, 91, 92), (100, 95, 96)),
    ((101, 97, 98), (102, 99, 100)),
    ((103, 101, 102),),
)

# --- FIFA World Cup 2018/2022 (32 teams, 8 groups) --------------------------
# Official match numbering: 49-56 round of 16, 57-60 quarter-finals,
# 61-62 semi-finals, 64 final (63 is the third-place play-off, which the
# simulator does not model because it never affects the champion).

ROUND_OF_16_MATCHES_32 = (
    (49, "W_A", "R_B"),
    (50, "W_C", "R_D"),
    (51, "W_B", "R_A"),
    (52, "W_D", "R_C"),
    (53, "W_E", "R_F"),
    (54, "W_G", "R_H"),
    (55, "W_F", "R_E"),
    (56, "W_H", "R_G"),
)

BRACKET_ROUNDS_32 = (
    ((57, 49, 50), (58, 53, 54), (59, 51, 52), (60, 55, 56)),
    ((61, 57, 58), (62, 59, 60)),
    ((64, 61, 62),),
)


@dataclass(frozen=True)
class ThirdPlaceRule:
    """Best-third-placed qualification (2026 edition)."""

    qualifier_count: int
    slot_allowed_groups: Mapping[str, frozenset[str]]


@dataclass(frozen=True)
class TournamentFormat:
    name: str
    group_count: int
    ranking_rules: str
    # Stage name that qualification from the group stage is equivalent to
    # reaching (kept as an explicit probability column for continuity).
    qualification_stage: str
    entry_matches: tuple[tuple[int, str, str], ...]
    # Stage reached by winning an entry-round match.
    entry_winners_reach: str
    bracket_rounds: tuple[tuple[tuple[int, int, int], ...], ...]
    # Stage reached by winning a match of bracket round i; the last round is
    # absent because winning the final is counted as "champion".
    stage_reached_by_bracket_round: Mapping[int, str]
    third_place: ThirdPlaceRule | None

    @property
    def final_match_id(self) -> int:
        return self.bracket_rounds[-1][0][0]

    @property
    def stage_columns(self) -> tuple[str, ...]:
        """Ordered probability columns produced by the simulators."""
        return (
            "group_qualify",
            self.qualification_stage,
            self.entry_winners_reach,
            *(
                self.stage_reached_by_bracket_round[index]
                for index in range(len(self.bracket_rounds) - 1)
            ),
            "champion",
        )

    @property
    def teams_reaching(self) -> dict[str, int]:
        """Exact number of teams reaching each stage in every simulation."""
        qualified = 2 * self.group_count + (
            self.third_place.qualifier_count if self.third_place else 0
        )
        counts = {
            "group_qualify": qualified,
            self.qualification_stage: qualified,
            self.entry_winners_reach: len(self.entry_matches),
        }
        for index in range(len(self.bracket_rounds) - 1):
            counts[self.stage_reached_by_bracket_round[index]] = len(
                self.bracket_rounds[index]
            )
        counts["champion"] = 1
        return counts


WC48_2026 = TournamentFormat(
    name="wc48_2026",
    group_count=12,
    ranking_rules="fifa_2026",
    qualification_stage="round_of_32",
    entry_matches=ROUND_OF_32_MATCHES,
    entry_winners_reach="round_of_16",
    bracket_rounds=BRACKET_ROUNDS_2026,
    stage_reached_by_bracket_round=MappingProxyType(
        {0: "quarter_final", 1: "semi_final", 2: "final"}
    ),
    third_place=ThirdPlaceRule(
        qualifier_count=8,
        slot_allowed_groups=MappingProxyType(THIRD_PLACE_SLOT_ALLOWED),
    ),
)

WC32 = TournamentFormat(
    name="wc32",
    group_count=8,
    ranking_rules="fifa_pre_2026",
    qualification_stage="round_of_16",
    entry_matches=ROUND_OF_16_MATCHES_32,
    entry_winners_reach="quarter_final",
    bracket_rounds=BRACKET_ROUNDS_32,
    stage_reached_by_bracket_round=MappingProxyType(
        {0: "semi_final", 1: "final"}
    ),
    third_place=None,
)

FORMATS: dict[str, TournamentFormat] = {
    WC48_2026.name: WC48_2026,
    WC32.name: WC32,
}


def get_format(name: str) -> TournamentFormat:
    try:
        return FORMATS[name]
    except KeyError:
        raise ValueError(
            f"Unknown tournament format {name!r}; available: {sorted(FORMATS)}"
        ) from None
