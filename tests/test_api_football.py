from __future__ import annotations

import httpx
from sqlalchemy import create_engine, select

from worldcup_predictor.ingestion.api_football import ApiFootballClient
from worldcup_predictor.store.db import (
    init_database,
    lineups,
    load_player_features_from_database,
    player_match_stats,
    sync_api_football_data,
)


def _response(payload: list[dict]) -> httpx.Response:
    return httpx.Response(200, json={"errors": {}, "response": payload})


def _handler(request: httpx.Request) -> httpx.Response:
    assert request.headers["x-apisports-key"] == "test-key"
    path = request.url.path
    if path == "/teams":
        return _response([
            {"team": {"id": 1, "name": "Atlas", "code": "ATL", "country": "Test"}},
            {"team": {"id": 2, "name": "Boreal", "code": "BOR", "country": "Test"}},
        ])
    if path == "/players/squads":
        team_id = request.url.params["team"]
        return _response([
            {
                "team": {"id": int(team_id)},
                "players": [
                    {
                        "id": 10 if team_id == "1" else 20,
                        "name": "Atlas One" if team_id == "1" else "Boreal One",
                        "position": "Attacker" if team_id == "1" else "Defender",
                    }
                ],
            }
        ])
    if path == "/injuries":
        return _response([
            {
                "player": {"id": 10, "type": "Injury", "reason": "Hamstring"},
                "team": {"id": 1},
            }
        ])
    if path == "/fixtures":
        return _response([
            {
                "fixture": {"id": 100, "date": "2026-06-11T18:00:00+00:00"},
                "teams": {"home": {"id": 1}, "away": {"id": 2}},
                "goals": {"home": 1, "away": 0},
            }
        ])
    if path == "/fixtures/lineups":
        return _response([
            {"team": {"id": 1}, "startXI": [{"player": {"id": 10}}], "substitutes": []},
            {"team": {"id": 2}, "startXI": [{"player": {"id": 20}}], "substitutes": []},
        ])
    if path == "/fixtures/players":
        return _response([
            {
                "team": {"id": 1},
                "players": [
                    {
                        "player": {"id": 10},
                        "statistics": [
                            {
                                "games": {"minutes": 90, "rating": "7.4"},
                                "goals": {"total": 1, "assists": 0},
                            }
                        ],
                    }
                ],
            },
            {
                "team": {"id": 2},
                "players": [
                    {
                        "player": {"id": 20},
                        "statistics": [
                            {
                                "games": {"minutes": 90, "rating": "6.8"},
                                "goals": {"total": 0, "assists": 0},
                            }
                        ],
                    }
                ],
            },
        ])
    raise AssertionError(f"Unexpected API path: {path}")


def test_api_football_sync_populates_player_hybrid_tables() -> None:
    client = ApiFootballClient(
        "test-key",
        base_url="https://api.example.test",
        client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    init_database(engine)

    summary = sync_api_football_data(engine, client, fixture_ids=[100])
    features = load_player_features_from_database(engine)

    assert summary["teams"] == 2
    assert summary["players"] == 2
    assert summary["injuries"] == 1
    assert summary["lineups"] == 2
    assert summary["player_match_stats"] == 2
    assert features.loc[features["player"] == "Atlas One", "available"].item() is False
    assert features.loc[
        features["player"] == "Boreal One", "defensive_rating"
    ].item() == 68.0

    with engine.connect() as connection:
        assert len(connection.execute(select(lineups.c.id)).all()) == 2
        assert len(connection.execute(select(player_match_stats.c.id)).all()) == 2
