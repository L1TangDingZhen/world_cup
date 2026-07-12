"""API-Football v3 client for live World Cup player data."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

API_FOOTBALL_PROVIDER = "api-football"
API_FOOTBALL_BASE_URL = "https://v3.football.api-sports.io"


class ApiFootballError(RuntimeError):
    """Raised for API-Football transport or payload errors."""


@dataclass(frozen=True)
class ApiFootballTeam:
    provider_id: int
    name: str
    code: str | None
    country: str | None


@dataclass(frozen=True)
class ApiFootballPlayer:
    provider_id: int
    name: str
    team_provider_id: int
    position: str | None


@dataclass(frozen=True)
class ApiFootballInjury:
    player_provider_id: int
    team_provider_id: int
    status: str
    reported_at: date


@dataclass(frozen=True)
class ApiFootballFixture:
    provider_id: int
    date: date
    home_team_provider_id: int
    away_team_provider_id: int
    home_goals: int | None
    away_goals: int | None


@dataclass(frozen=True)
class ApiFootballLineupEntry:
    player_provider_id: int
    team_provider_id: int
    starter: bool


@dataclass(frozen=True)
class ApiFootballPlayerStat:
    player_provider_id: int
    team_provider_id: int
    minutes_played: int | None
    goals: int | None
    assists: int | None
    rating: float | None


class ApiFootballClient:
    """Small synchronous client with an injectable httpx client for tests."""

    def __init__(
        self,
        api_key: str,
        base_url: str = API_FOOTBALL_BASE_URL,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("API-Football API key is required")
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=30.0)
        self._owns_client = client is None
        self._headers = {"x-apisports-key": api_key}

    @classmethod
    def from_environment(cls, api_key: str | None = None) -> "ApiFootballClient":
        return cls(api_key or os.getenv("API_FOOTBALL_KEY", ""))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "ApiFootballClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _get(self, path: str, params: dict[str, object]) -> list[dict[str, Any]]:
        response = self._client.get(
            f"{self.base_url}/{path.lstrip('/')}",
            params=params,
            headers=self._headers,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ApiFootballError(f"API-Football request failed for {path}: {exc}") from exc
        payload = response.json()
        errors = payload.get("errors", {})
        if errors:
            raise ApiFootballError(f"API-Football returned errors for {path}: {errors}")
        rows = payload.get("response")
        if not isinstance(rows, list):
            raise ApiFootballError(f"API-Football response for {path} has no response list")
        return rows

    def world_cup_teams(self, league: int = 1, season: int = 2026) -> list[ApiFootballTeam]:
        rows = self._get("teams", {"league": league, "season": season})
        return [
            ApiFootballTeam(
                provider_id=int(row["team"]["id"]),
                name=str(row["team"]["name"]),
                code=_optional_text(row["team"].get("code")),
                country=_optional_text(row["team"].get("country")),
            )
            for row in rows
        ]

    def squad(self, team_provider_id: int) -> list[ApiFootballPlayer]:
        rows = self._get("players/squads", {"team": team_provider_id})
        players: list[ApiFootballPlayer] = []
        for row in rows:
            team_id = int(row.get("team", {}).get("id", team_provider_id))
            for player in row.get("players", []):
                players.append(
                    ApiFootballPlayer(
                        provider_id=int(player["id"]),
                        name=str(player["name"]),
                        team_provider_id=team_id,
                        position=_optional_text(player.get("position")),
                    )
                )
        return players

    def injuries(
        self,
        league: int = 1,
        season: int = 2026,
        as_of: date | None = None,
    ) -> list[ApiFootballInjury]:
        rows = self._get("injuries", {"league": league, "season": season})
        reported_at = as_of or date.today()
        injuries: list[ApiFootballInjury] = []
        for row in rows:
            player = row.get("player", {})
            team = row.get("team", {})
            if player.get("id") is None or team.get("id") is None:
                continue
            status_parts = [row.get("player", {}).get("type"), row.get("player", {}).get("reason")]
            status = " - ".join(str(part) for part in status_parts if part) or "Unavailable"
            injuries.append(
                ApiFootballInjury(
                    player_provider_id=int(player["id"]),
                    team_provider_id=int(team["id"]),
                    status=status,
                    reported_at=reported_at,
                )
            )
        return injuries

    def fixture(self, fixture_provider_id: int) -> ApiFootballFixture:
        rows = self._get("fixtures", {"id": fixture_provider_id})
        if not rows:
            raise ApiFootballError(f"Fixture {fixture_provider_id} was not found")
        row = rows[0]
        fixture = row["fixture"]
        teams = row["teams"]
        goals = row.get("goals", {})
        return ApiFootballFixture(
            provider_id=int(fixture["id"]),
            date=date.fromisoformat(str(fixture["date"])[:10]),
            home_team_provider_id=int(teams["home"]["id"]),
            away_team_provider_id=int(teams["away"]["id"]),
            home_goals=_optional_int(goals.get("home")),
            away_goals=_optional_int(goals.get("away")),
        )

    def fixture_lineup(self, fixture_provider_id: int) -> list[ApiFootballLineupEntry]:
        rows = self._get("fixtures/lineups", {"fixture": fixture_provider_id})
        entries: list[ApiFootballLineupEntry] = []
        for row in rows:
            team_id = int(row["team"]["id"])
            for item in row.get("startXI", []):
                entries.append(
                    ApiFootballLineupEntry(
                        player_provider_id=int(item["player"]["id"]),
                        team_provider_id=team_id,
                        starter=True,
                    )
                )
            for item in row.get("substitutes", []):
                entries.append(
                    ApiFootballLineupEntry(
                        player_provider_id=int(item["player"]["id"]),
                        team_provider_id=team_id,
                        starter=False,
                    )
                )
        return entries

    def fixture_player_stats(self, fixture_provider_id: int) -> list[ApiFootballPlayerStat]:
        rows = self._get("fixtures/players", {"fixture": fixture_provider_id})
        stats: list[ApiFootballPlayerStat] = []
        for row in rows:
            team_id = int(row["team"]["id"])
            for item in row.get("players", []):
                details = (item.get("statistics") or [{}])[0]
                games = details.get("games", {})
                goals = details.get("goals", {})
                stats.append(
                    ApiFootballPlayerStat(
                        player_provider_id=int(item["player"]["id"]),
                        team_provider_id=team_id,
                        minutes_played=_optional_int(games.get("minutes")),
                        goals=_optional_int(goals.get("total")),
                        assists=_optional_int(goals.get("assists")),
                        rating=_optional_float(games.get("rating")),
                    )
                )
        return stats


def _optional_text(value: object) -> str | None:
    return str(value) if value not in {None, ""} else None


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_float(value: object) -> float | None:
    return float(value) if value not in {None, ""} else None
