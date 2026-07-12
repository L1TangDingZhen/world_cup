from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from worldcup_predictor.compute import (
    ComputeDevice,
    DeviceUnavailableError,
    resolve_device,
)
from worldcup_predictor.models.elo_poisson import EloPoissonModel
from worldcup_predictor.simulation.actual_results import (
    load_knockout_winners_from_files,
)
from worldcup_predictor.simulation.tournament import (
    TournamentConfig,
    TournamentSimulator,
)
from worldcup_predictor.tasks import dynamic_update_task
from worldcup_predictor.workflows.catch_up import catch_up
from worldcup_predictor.workflows.dynamic_update import (
    MatchResultInput,
    run_dynamic_update,
)

MODEL_PATH = Path(os.getenv("WORLDCUP_MODEL_PATH", "models/elo_poisson_current.json"))
GROUPS_PATH = Path(os.getenv("WORLDCUP_GROUPS_PATH", "data/worldcup/groups_2026.csv"))
FIXTURES_PATH = Path(os.getenv("WORLDCUP_FIXTURES_PATH", "data/worldcup/fixtures_2026.csv"))
MATCHES_PATH = Path(
    os.getenv("WORLDCUP_MATCHES_PATH", "data/raw/international_results.csv")
)
SHOOTOUTS_PATH = Path(os.getenv("WORLDCUP_SHOOTOUTS_PATH", "data/raw/shootouts.csv"))

app = FastAPI(title="World Cup Predictor API", version="0.1.0")


class SimulationRequest(BaseModel):
    simulations: int = Field(default=1000, ge=1, le=100_000)
    seed: int | None = None
    device: ComputeDevice = "auto"
    # Download the latest results, fill fixtures and refit before simulating.
    sync: bool = False
    # Pin knockout matches already played to their real winners.
    condition_knockouts: bool = True


class MatchResultRequest(BaseModel):
    date: str
    home_team: str
    away_team: str
    home_goals: int = Field(ge=0)
    away_goals: int = Field(ge=0)
    competition_type: str = "FIFA World Cup"
    neutral_venue: bool = True
    # The refit uses this file as the FULL training set, so it must be the
    # complete match history, not a side file with a handful of results.
    matches_path: str = "data/raw/international_results.csv"
    simulations: int = Field(default=500, ge=1, le=20_000)


@lru_cache(maxsize=1)
def get_model() -> EloPoissonModel:
    return EloPoissonModel.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_config() -> TournamentConfig:
    return TournamentConfig.from_csv(GROUPS_PATH, FIXTURES_PATH)


@lru_cache(maxsize=1)
def get_knockout_winners() -> dict[frozenset[str], str] | None:
    if not MATCHES_PATH.is_file():
        return None
    return load_knockout_winners_from_files(
        raw_path=MATCHES_PATH,
        config=get_config(),
        shootouts_path=SHOOTOUTS_PATH,
    )


def clear_caches() -> None:
    get_model.cache_clear()
    get_config.cache_clear()
    get_knockout_winners.cache_clear()


def resolve_request_device(device: ComputeDevice) -> str:
    try:
        return resolve_device(device).name
    except DeviceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/teams")
async def teams(limit: int = Query(default=100, ge=1, le=500)) -> list[dict[str, float | str | int]]:
    return [
        {"rank": index, "team": team, "elo": rating}
        for index, (team, rating) in enumerate(get_model().rankings()[:limit], start=1)
    ]


@app.get("/matches")
async def matches() -> list[dict[str, object]]:
    fixtures = get_config().fixtures.copy()
    fixtures["date"] = fixtures["date"].dt.date.astype(str)
    return fixtures.to_dict(orient="records")


@app.get("/predict")
async def predict(
    home: str,
    away: str,
    neutral: bool = True,
    include_score_matrix: bool = False,
    device: ComputeDevice = "auto",
) -> dict[str, object]:
    resolved_device = resolve_request_device(device)
    try:
        prediction = get_model().predict(
            home,
            away,
            neutral_venue=neutral,
            device=resolved_device,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    payload = prediction.to_dict(include_score_matrix=include_score_matrix)
    payload["device"] = resolved_device
    return payload


@app.get("/tournament/group-stage")
async def group_stage() -> list[dict[str, object]]:
    return get_config().groups.sort_values(["group", "team"]).to_dict(orient="records")


@app.get("/tournament/champion-probabilities")
async def champion_probabilities(
    simulations: int = Query(default=1000, ge=1, le=100_000),
    seed: int | None = None,
    device: ComputeDevice = "auto",
    condition_knockouts: bool = True,
) -> list[dict[str, object]]:
    resolved_device = resolve_request_device(device)
    simulator = TournamentSimulator(
        get_model(),
        get_config(),
        random_seed=seed,
        device=resolved_device,
        knockout_winners=get_knockout_winners() if condition_knockouts else None,
    )
    result = simulator.run(simulations=simulations)
    return result[["team", "champion_prob"]].to_dict(orient="records")


@app.post("/simulation/run")
async def simulation_run(request: SimulationRequest) -> list[dict[str, object]]:
    resolved_device = resolve_request_device(request.device)
    if request.sync:
        catch_up(
            raw_path=MATCHES_PATH,
            fixtures_path=FIXTURES_PATH,
            model_output=MODEL_PATH,
            shootouts_path=SHOOTOUTS_PATH,
        )
        clear_caches()
    simulator = TournamentSimulator(
        get_model(),
        get_config(),
        random_seed=request.seed,
        device=resolved_device,
        knockout_winners=(
            get_knockout_winners() if request.condition_knockouts else None
        ),
    )
    result = simulator.run(simulations=request.simulations)
    return result.to_dict(orient="records")


@app.post("/matches/result")
async def record_result(request: MatchResultRequest) -> dict[str, object]:
    result = MatchResultInput(
        date=request.date,
        home_team=request.home_team,
        away_team=request.away_team,
        home_goals=request.home_goals,
        away_goals=request.away_goals,
        competition_type=request.competition_type,
        neutral_venue=request.neutral_venue,
    )
    if os.getenv("CELERY_ASYNC_ENABLED", "false").lower() in {"1", "true", "yes"}:
        task = dynamic_update_task.delay(
            request.matches_path,
            result.__dict__,
            {"simulations": request.simulations},
        )
        return {"status": "queued", "task_id": task.id}
    summary = run_dynamic_update(
        matches_path=request.matches_path,
        result=result,
        simulations=request.simulations,
    )
    clear_caches()
    return summary
