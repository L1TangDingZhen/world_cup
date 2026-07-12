from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, inspect, select, text

from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.store.db import (
    init_database,
    load_matches_dataframe,
    matches,
    models,
    predictions,
    persist_elo_model,
    persist_prediction,
    persist_simulation_results,
    simulation_results,
    table_names,
    teams,
    write_schema_sql,
)
from worldcup_predictor.models import EloPoissonModel


def test_db_schema_and_match_load(tmp_path) -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    init_database(engine)

    frame = load_matches("data/examples/synthetic_matches.csv")
    summary = load_matches_dataframe(engine, frame)

    with engine.connect() as connection:
        team_count = connection.execute(select(teams.c.id)).all()
        match_count = connection.execute(select(matches.c.id)).all()

    assert summary == {"teams": 4, "matches": 18}
    assert len(team_count) == 4
    assert len(match_count) == 18
    assert {
        "teams", "matches", "models", "rating_snapshots", "players", "squads",
        "player_match_stats", "injuries", "lineups",
    } <= set(table_names())

    schema_path = write_schema_sql(tmp_path / "schema.sql", database_url="sqlite://")
    schema_text = schema_path.read_text(encoding="utf-8")
    assert "CREATE TABLE teams" in schema_text
    assert "CREATE TABLE simulation_results" in schema_text


def test_db_persists_model_prediction_and_simulation() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    init_database(engine)
    model = EloPoissonModel().fit(load_matches("data/examples/synthetic_matches.csv"))
    prediction = model.predict("Atlas", "Comet")
    simulation = pd.DataFrame(
        [{
            "team": "Atlas", "group_qualify_prob": 1.0, "round_of_32_prob": 1.0,
            "round_of_16_prob": 0.8, "quarter_final_prob": 0.5,
            "semi_final_prob": 0.3, "final_prob": 0.2, "champion_prob": 0.1,
        }]
    )

    persist_elo_model(engine, model)
    persist_prediction(engine, prediction, model.model_version)
    persist_simulation_results(engine, simulation, "test-run", model.model_version)

    with engine.connect() as connection:
        assert connection.execute(select(models.c.id)).first() is not None
        assert connection.execute(select(predictions.c.id)).first() is not None
        assert connection.execute(select(simulation_results.c.id)).first() is not None


def test_init_database_upgrades_legacy_player_data_tables(tmp_path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'legacy.db'}", future=True)
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE TABLE teams ("
            "id INTEGER PRIMARY KEY, name VARCHAR(120) NOT NULL, "
            "code VARCHAR(12), confederation VARCHAR(32), "
            "created_at DATETIME, updated_at DATETIME)"
        ))
        connection.execute(text(
            "CREATE TABLE players ("
            "id INTEGER PRIMARY KEY, name VARCHAR(120) NOT NULL, "
            "national_team_id INTEGER, position VARCHAR(32), created_at DATETIME)"
        ))
        connection.execute(text(
            "CREATE TABLE matches ("
            "id INTEGER PRIMARY KEY, date DATE NOT NULL, "
            "home_team_id INTEGER NOT NULL, away_team_id INTEGER NOT NULL, "
            "home_goals INTEGER, away_goals INTEGER, competition_type VARCHAR(120) NOT NULL, "
            "country VARCHAR(120), neutral_venue BOOLEAN NOT NULL, "
            "created_at DATETIME, updated_at DATETIME)"
        ))

    init_database(engine)

    inspector = inspect(engine)
    assert {"provider", "provider_id"} <= {
        column["name"] for column in inspector.get_columns("teams")
    }
    assert {"provider", "provider_id"} <= {
        column["name"] for column in inspector.get_columns("players")
    }
    assert {"provider", "provider_id"} <= {
        column["name"] for column in inspector.get_columns("matches")
    }
    assert "rating" in {
        column["name"] for column in inspector.get_columns("player_match_stats")
    }
    simulation_results,
