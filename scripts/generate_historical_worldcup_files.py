"""Generate groups/fixtures/actual-progress files for the 2018 and 2022 World Cups.

The group-stage fixture list only needs the correct pairings, venue flags and
a nominal date: every group plays a single round robin, match order does not
affect the simulation, and with neutral_venue=True the home/away orientation
is symmetric for the models. Host-nation matches carry home advantage, so
the host is oriented as the home team with neutral_venue=false.

Team names follow the martj42/international_results conventions, matching
the historical training data.
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

GROUPS_2018 = {
    "A": ["Russia", "Saudi Arabia", "Egypt", "Uruguay"],
    "B": ["Portugal", "Spain", "Morocco", "Iran"],
    "C": ["France", "Australia", "Peru", "Denmark"],
    "D": ["Argentina", "Iceland", "Croatia", "Nigeria"],
    "E": ["Brazil", "Switzerland", "Costa Rica", "Serbia"],
    "F": ["Germany", "Mexico", "Sweden", "South Korea"],
    "G": ["Belgium", "Panama", "Tunisia", "England"],
    "H": ["Poland", "Senegal", "Colombia", "Japan"],
}

GROUPS_2022 = {
    "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
    "B": ["England", "Iran", "United States", "Wales"],
    "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
    "D": ["France", "Australia", "Denmark", "Tunisia"],
    "E": ["Spain", "Costa Rica", "Germany", "Japan"],
    "F": ["Belgium", "Canada", "Morocco", "Croatia"],
    "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
    "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
}

# Furthest stage reached, in "reach" semantics (losing the semi-final still
# means the semi-final was reached; the third-place play-off is ignored).
ACTUAL_2018 = {
    "champion": ["France"],
    "final": ["Croatia"],
    "semi_final": ["Belgium", "England"],
    "quarter_final": ["Uruguay", "Russia", "Brazil", "Sweden"],
    "round_of_16": [
        "Portugal", "Spain", "Denmark", "Argentina",
        "Switzerland", "Mexico", "Colombia", "Japan",
    ],
    "group": [
        "Saudi Arabia", "Egypt", "Morocco", "Iran", "Australia", "Peru",
        "Iceland", "Nigeria", "Costa Rica", "Serbia", "Germany",
        "South Korea", "Panama", "Tunisia", "Poland", "Senegal",
    ],
}

ACTUAL_2022 = {
    "champion": ["Argentina"],
    "final": ["France"],
    "semi_final": ["Croatia", "Morocco"],
    "quarter_final": ["Netherlands", "England", "Brazil", "Portugal"],
    "round_of_16": [
        "Senegal", "United States", "Poland", "Australia",
        "Japan", "Spain", "Switzerland", "South Korea",
    ],
    "group": [
        "Qatar", "Ecuador", "Iran", "Wales", "Saudi Arabia", "Mexico",
        "Denmark", "Tunisia", "Costa Rica", "Germany", "Canada", "Belgium",
        "Serbia", "Cameroon", "Ghana", "Uruguay",
    ],
}

EDITIONS = {
    "2018": {
        "groups": GROUPS_2018,
        "actual": ACTUAL_2018,
        "host": "Russia",
        "start_date": "2018-06-14",
    },
    "2022": {
        "groups": GROUPS_2022,
        "actual": ACTUAL_2022,
        "host": "Qatar",
        "start_date": "2022-11-20",
    },
}


def write_edition(year: str, output_dir: Path) -> None:
    edition = EDITIONS[year]
    groups = edition["groups"]
    host = edition["host"]
    start_date = edition["start_date"]

    groups_path = output_dir / f"groups_{year}.csv"
    with groups_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "team"])
        for group in sorted(groups):
            for team in groups[group]:
                writer.writerow([group, team])

    fixtures_path = output_dir / f"fixtures_{year}.csv"
    with fixtures_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["group", "date", "home_team", "away_team",
             "home_goals", "away_goals", "neutral_venue"]
        )
        for group in sorted(groups):
            for team_a, team_b in combinations(groups[group], 2):
                if team_b == host:
                    team_a, team_b = team_b, team_a
                neutral = "false" if team_a == host else "true"
                writer.writerow([group, start_date, team_a, team_b, "", "", neutral])

    actual_path = output_dir / f"actual_{year}.csv"
    with actual_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["team", "furthest"])
        for furthest, teams in edition["actual"].items():
            for team in teams:
                writer.writerow([team, furthest])

    all_teams = {team for members in groups.values() for team in members}
    listed = [team for teams in edition["actual"].values() for team in teams]
    if sorted(listed) != sorted(all_teams):
        raise AssertionError(f"{year}: actual progress does not cover the 32 teams exactly")
    print(f"{year}: wrote {groups_path.name}, {fixtures_path.name}, {actual_path.name}")


def main() -> None:
    output_dir = Path("data/worldcup")
    output_dir.mkdir(parents=True, exist_ok=True)
    for year in EDITIONS:
        write_edition(year, output_dir)


if __name__ == "__main__":
    main()
