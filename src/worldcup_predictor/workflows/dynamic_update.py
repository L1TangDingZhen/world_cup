from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.models.elo_poisson import EloPoissonModel
from worldcup_predictor.simulation.tournament import (
    TournamentConfig,
    TournamentSimulator,
)


@dataclass(frozen=True)
class MatchResultInput:
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    competition_type: str
    neutral_venue: bool


@dataclass(frozen=True)
class DynamicUpdateSummary:
    matches: int
    model_output: str
    simulation_output: str
    predictions_output: str
    simulations: int
    upserted_existing_result: bool
    updated_fixture: bool


def append_result(
    csv_path: str | Path,
    result: MatchResultInput,
    replace_existing: bool = True,
) -> bool:
    path = Path(csv_path)
    canonical_row = asdict(result)
    if path.exists():
        existing = pd.read_csv(path)
        if "home_score" in existing.columns:
            row_values = {
                "date": result.date,
                "home_team": result.home_team,
                "away_team": result.away_team,
                "home_score": result.home_goals,
                "away_score": result.away_goals,
                "tournament": result.competition_type,
                "neutral": result.neutral_venue,
            }
        else:
            row_values = canonical_row
        row = pd.DataFrame([{column: row_values.get(column) for column in existing.columns}])
        competition_column = "tournament" if "tournament" in existing.columns else "competition_type"
        mask = (
            (existing["date"].astype(str) == result.date)
            & (existing["home_team"].astype(str) == result.home_team)
            & (existing["away_team"].astype(str) == result.away_team)
            & (existing[competition_column].astype(str) == result.competition_type)
        )
        upserted = bool(mask.any())
        if upserted and replace_existing:
            output = existing.loc[~mask].copy()
            output = pd.concat([output, row], ignore_index=True)
        elif upserted:
            output = existing
        else:
            output = pd.concat([existing, row], ignore_index=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        row = pd.DataFrame([canonical_row])
        output = row
        upserted = False
    output.to_csv(path, index=False)
    return upserted


def update_fixture_result(fixtures_path: str | Path, result: MatchResultInput) -> bool:
    path = Path(fixtures_path)
    if not path.exists():
        return False
    fixtures = pd.read_csv(path)
    mask = (
        (fixtures["date"].astype(str) == result.date)
        & (fixtures["home_team"].astype(str) == result.home_team)
        & (fixtures["away_team"].astype(str) == result.away_team)
    )
    if not mask.any():
        return False
    fixtures.loc[mask, "home_goals"] = result.home_goals
    fixtures.loc[mask, "away_goals"] = result.away_goals
    fixtures.to_csv(path, index=False)
    return True


def predict_remaining_fixtures(
    model: EloPoissonModel,
    fixtures_path: str | Path,
    output_path: str | Path,
) -> int:
    fixtures = pd.read_csv(fixtures_path)
    remaining = fixtures.loc[
        fixtures["home_goals"].isna() | fixtures["away_goals"].isna()
    ].copy()
    rows = []
    for fixture in remaining.itertuples(index=False):
        prediction = model.predict(
            fixture.home_team,
            fixture.away_team,
            neutral_venue=bool(fixture.neutral_venue),
        )
        rows.append(prediction.to_dict(include_score_matrix=False) | {"date": fixture.date})
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return len(rows)


def run_dynamic_update(
    matches_path: str | Path,
    result: MatchResultInput,
    model_output: str | Path = "models/elo_poisson_current.json",
    simulation_output: str | Path = "data/processed/simulation_2026.csv",
    predictions_output: str | Path = "data/processed/remaining_predictions.csv",
    groups_path: str | Path = "data/worldcup/groups_2026.csv",
    fixtures_path: str | Path = "data/worldcup/fixtures_2026.csv",
    simulations: int = 1000,
    seed: int = 42,
    min_training_matches: int = 1000,
) -> dict[str, object]:
    upserted = append_result(matches_path, result)
    updated_fixture = update_fixture_result(fixtures_path, result)
    matches = load_matches(matches_path, completed_only=True)
    if len(matches) < min_training_matches:
        raise ValueError(
            f"Refusing to refit from {matches_path}: it has {len(matches)} matches "
            f"but at least {min_training_matches} are required. The refit replaces "
            f"{model_output}, so matches_path must point at the full match history."
        )
    model = EloPoissonModel().fit(matches)
    model.save(model_output)
    predict_remaining_fixtures(model, fixtures_path, predictions_output)

    simulation = TournamentSimulator(
        model,
        TournamentConfig.from_csv(groups_path, fixtures_path),
        random_seed=seed,
    ).run(simulations=simulations)
    simulation_path = Path(simulation_output)
    simulation_path.parent.mkdir(parents=True, exist_ok=True)
    simulation.to_csv(simulation_path, index=False)
    return asdict(
        DynamicUpdateSummary(
            matches=len(matches),
            model_output=str(model_output),
            simulation_output=str(simulation_path),
            predictions_output=str(predictions_output),
            simulations=simulations,
            upserted_existing_result=upserted,
            updated_fixture=updated_fixture,
        )
    )
