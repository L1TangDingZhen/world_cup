from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from worldcup_predictor.api.main import app


@pytest.mark.anyio
async def test_health_and_teams() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        assert (await client.get("/health")).json() == {"status": "ok"}
        teams = (await client.get("/teams", params={"limit": 3})).json()

    assert len(teams) == 3
    assert {"rank", "team", "elo"} <= set(teams[0])


@pytest.mark.anyio
async def test_predict_endpoint() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/predict",
            params={"home": "Argentina", "away": "France", "neutral": True},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["most_likely_score"]
    assert payload["device"] in {"cpu", "cuda"}
    assert abs(
        payload["home_win_prob"] + payload["draw_prob"] + payload["away_win_prob"] - 1
    ) < 1e-9


@pytest.mark.anyio
async def test_explicit_cuda_returns_service_unavailable_without_cuda() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/predict",
            params={
                "home": "Argentina",
                "away": "France",
                "neutral": True,
                "device": "cuda",
            },
        )

    if response.status_code == 200:
        assert response.json()["device"] == "cuda"
    else:
        assert response.status_code == 503


@pytest.mark.anyio
async def test_simulation_endpoint() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            "/simulation/run",
            json={"simulations": 2, "seed": 123},
        )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 48
    assert "champion_prob" in payload[0]
