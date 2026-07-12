from __future__ import annotations

import os

from celery import Celery

from worldcup_predictor.ingestion.api_football import ApiFootballClient
from worldcup_predictor.store.db import engine_from_url, sync_api_football_data
from worldcup_predictor.workflows.dynamic_update import (
    MatchResultInput,
    run_dynamic_update,
)

celery_app = Celery(
    "worldcup_predictor",
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),
)
celery_app.conf.beat_schedule = {
    "sync-api-football-player-data": {
        "task": "worldcup_predictor.sync_api_football",
        "schedule": 6 * 60 * 60,
    }
}


@celery_app.task(name="worldcup_predictor.dynamic_update")
def dynamic_update_task(
    matches_path: str,
    result: dict[str, object],
    options: dict[str, object] | None = None,
) -> dict[str, object]:
    options = options or {}
    return run_dynamic_update(
        matches_path=matches_path,
        result=MatchResultInput(**result),
        **options,
    )


@celery_app.task(name="worldcup_predictor.sync_api_football")
def sync_api_football_task(
    database_url: str | None = None,
    league: int = 1,
    season: int = 2026,
    fixture_ids: list[int] | None = None,
) -> dict[str, object]:
    if os.getenv("API_FOOTBALL_SYNC_ENABLED", "false").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return {
            "status": "skipped",
            "reason": "Set API_FOOTBALL_SYNC_ENABLED=true to enable provider sync.",
        }
    with ApiFootballClient.from_environment() as client:
        return sync_api_football_data(
            engine_from_url(
                database_url
                or os.getenv(
                    "WORLDCUP_DATABASE_URL",
                    "postgresql+psycopg://worldcup:worldcup@postgres:5432/worldcup",
                )
            ),
            client,
            league=league,
            season=season,
            fixture_ids=fixture_ids,
        )
