from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    create_mock_engine,
    delete,
    func,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine

from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.ingestion.api_football import (
    API_FOOTBALL_PROVIDER,
    ApiFootballClient,
)
from worldcup_predictor.models.elo_poisson import EloPoissonModel, MatchPrediction

metadata = MetaData()

teams = Table(
    "teams",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(120), nullable=False, unique=True),
    Column("code", String(12)),
    Column("confederation", String(32)),
    Column("provider", String(32)),
    Column("provider_id", String(64)),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

players = Table(
    "players",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("name", String(120), nullable=False),
    Column("national_team_id", ForeignKey("teams.id")),
    Column("position", String(32)),
    Column("provider", String(32)),
    Column("provider_id", String(64)),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

squads = Table(
    "squads",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("team_id", ForeignKey("teams.id"), nullable=False),
    Column("player_id", ForeignKey("players.id"), nullable=False),
    Column("as_of_date", Date, nullable=False),
    Column("available", Boolean, nullable=False, default=True),
    Column("attacking_rating", Float),
    Column("defensive_rating", Float),
)

player_match_stats = Table(
    "player_match_stats",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("player_id", ForeignKey("players.id"), nullable=False),
    Column("match_id", ForeignKey("matches.id"), nullable=False),
    Column("minutes_played", Integer),
    Column("goals", Integer),
    Column("assists", Integer),
    Column("xg", Float),
    Column("xa", Float),
    Column("rating", Float),
)

injuries = Table(
    "injuries",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("player_id", ForeignKey("players.id"), nullable=False),
    Column("reported_at", Date, nullable=False),
    Column("status", String(80), nullable=False),
    Column("expected_return", Date),
)

lineups = Table(
    "lineups",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("match_id", ForeignKey("matches.id"), nullable=False),
    Column("player_id", ForeignKey("players.id"), nullable=False),
    Column("team_id", ForeignKey("teams.id"), nullable=False),
    Column("starter", Boolean, nullable=False),
)

matches = Table(
    "matches",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("date", Date, nullable=False),
    Column("home_team_id", ForeignKey("teams.id"), nullable=False),
    Column("away_team_id", ForeignKey("teams.id"), nullable=False),
    Column("home_goals", Integer),
    Column("away_goals", Integer),
    Column("competition_type", String(120), nullable=False),
    Column("country", String(120)),
    Column("provider", String(32)),
    Column("provider_id", String(64)),
    Column("neutral_venue", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("updated_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

models = Table(
    "models",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("model_version", String(80), nullable=False, unique=True),
    Column("model_type", String(80), nullable=False),
    Column("home_advantage_gamma", Float),
    Column("rho", Float),
    Column("time_decay_xi", Float),
    Column("competition_weights_json", JSON),
    Column("elo_k_factor", Float),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

rating_snapshots = Table(
    "rating_snapshots",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("computed_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
    Column("team_id", ForeignKey("teams.id"), nullable=False),
    Column("model_version", String(80), nullable=False),
    Column("elo", Float),
    Column("attack", Float),
    Column("defense", Float),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

predictions = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("match_id", ForeignKey("matches.id")),
    Column("home_team_id", ForeignKey("teams.id"), nullable=False),
    Column("away_team_id", ForeignKey("teams.id"), nullable=False),
    Column("home_win_prob", Float, nullable=False),
    Column("draw_prob", Float, nullable=False),
    Column("away_win_prob", Float, nullable=False),
    Column("most_likely_score", String(16), nullable=False),
    Column("score_matrix_json", JSON),
    Column("model_version", String(80), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)

simulation_results = Table(
    "simulation_results",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("simulation_run_id", String(80), nullable=False),
    Column("team_id", ForeignKey("teams.id"), nullable=False),
    Column("group_qualify_prob", Float, nullable=False),
    Column("round_of_32_prob", Float, nullable=False),
    Column("round_of_16_prob", Float, nullable=False),
    Column("quarter_final_prob", Float, nullable=False),
    Column("semi_final_prob", Float, nullable=False),
    Column("final_prob", Float, nullable=False),
    Column("champion_prob", Float, nullable=False),
    Column("model_version", String(80), nullable=False),
    Column("created_at", DateTime(timezone=True), server_default=func.now(), nullable=False),
)


def engine_from_url(database_url: str) -> Engine:
    return create_engine(database_url, future=True)


def init_database(engine: Engine) -> None:
    metadata.create_all(engine)
    _add_missing_provider_columns(engine)


def _add_missing_provider_columns(engine: Engine) -> None:
    """Keep pre-existing Step 6 databases compatible with player data sync."""
    required = {
        "teams": ("provider", "provider_id"),
        "players": ("provider", "provider_id"),
        "matches": ("provider", "provider_id"),
        "player_match_stats": ("rating",),
    }
    inspector = inspect(engine)
    with engine.begin() as connection:
        for table_name, columns in required.items():
            existing = {
                column["name"] for column in inspector.get_columns(table_name)
            }
            table = metadata.tables[table_name]
            for column_name in columns:
                if column_name in existing:
                    continue
                type_sql = table.c[column_name].type.compile(dialect=engine.dialect)
                connection.execute(
                    text(
                        f'ALTER TABLE "{table_name}" '
                        f'ADD COLUMN "{column_name}" {type_sql}'
                    )
                )


def table_names() -> list[str]:
    return sorted(metadata.tables)


def load_matches_dataframe(engine: Engine, frame: pd.DataFrame) -> dict[str, int]:
    unique_teams = sorted(set(frame["home_team"]) | set(frame["away_team"]))
    with engine.begin() as connection:
        existing = {
            name
            for name in connection.execute(select(teams.c.name)).scalars().all()
        }
        new_teams = [{"name": team} for team in unique_teams if team not in existing]
        if new_teams:
            connection.execute(teams.insert(), new_teams)

        team_ids = {
            row.name: row.id
            for row in connection.execute(select(teams.c.id, teams.c.name))
        }
        rows = [
            {
                "date": match.date.date(),
                "home_team_id": team_ids[match.home_team],
                "away_team_id": team_ids[match.away_team],
                "home_goals": int(match.home_goals),
                "away_goals": int(match.away_goals),
                "competition_type": match.competition_type,
                "neutral_venue": bool(match.neutral_venue),
            }
            for match in frame.itertuples(index=False)
        ]
        if rows:
            connection.execute(matches.insert(), rows)
    return {"teams": len(unique_teams), "matches": len(frame)}


def load_matches_csv(
    engine: Engine,
    path: str | Path,
    completed_only: bool = True,
) -> dict[str, int]:
    return load_matches_dataframe(
        engine,
        load_matches(path, completed_only=completed_only),
    )


def _ensure_teams(connection, names: list[str]) -> dict[str, int]:
    existing = {name for name in connection.execute(select(teams.c.name)).scalars().all()}
    new_rows = [{"name": name} for name in sorted(set(names) - existing)]
    if new_rows:
        connection.execute(teams.insert(), new_rows)
    return {row.name: row.id for row in connection.execute(select(teams.c.id, teams.c.name))}


def _upsert_api_football_teams(connection, remote_teams) -> dict[int, int]:
    existing_by_provider = {
        str(row.provider_id): row.id
        for row in connection.execute(
            select(teams.c.id, teams.c.provider_id).where(
                teams.c.provider == API_FOOTBALL_PROVIDER
            )
        )
    }
    existing_by_name = {
        row.name: row.id
        for row in connection.execute(select(teams.c.id, teams.c.name))
    }
    team_ids: dict[int, int] = {}
    for remote in remote_teams:
        values = {
            "name": remote.name,
            "code": remote.code,
            "confederation": remote.country,
            "provider": API_FOOTBALL_PROVIDER,
            "provider_id": str(remote.provider_id),
        }
        existing_id = existing_by_provider.get(str(remote.provider_id))
        if existing_id is None:
            existing_id = existing_by_name.get(remote.name)
        if existing_id is None:
            result = connection.execute(teams.insert().values(**values))
            existing_id = int(result.inserted_primary_key[0])
        else:
            connection.execute(
                update(teams).where(teams.c.id == existing_id).values(**values)
            )
        team_ids[remote.provider_id] = existing_id
    return team_ids


def _upsert_api_football_players(connection, remote_players, team_ids: dict[int, int]) -> dict[int, int]:
    existing = {
        str(row.provider_id): row.id
        for row in connection.execute(
            select(players.c.id, players.c.provider_id).where(
                players.c.provider == API_FOOTBALL_PROVIDER
            )
        )
    }
    player_ids: dict[int, int] = {}
    for remote in remote_players:
        team_id = team_ids.get(remote.team_provider_id)
        if team_id is None:
            continue
        values = {
            "name": remote.name,
            "national_team_id": team_id,
            "position": remote.position,
            "provider": API_FOOTBALL_PROVIDER,
            "provider_id": str(remote.provider_id),
        }
        existing_id = existing.get(str(remote.provider_id))
        if existing_id is None:
            result = connection.execute(players.insert().values(**values))
            existing_id = int(result.inserted_primary_key[0])
            existing[str(remote.provider_id)] = existing_id
        else:
            connection.execute(
                update(players).where(players.c.id == existing_id).values(**values)
            )
        player_ids[remote.provider_id] = existing_id
    return player_ids


def _upsert_api_football_match(connection, fixture, team_ids: dict[int, int]) -> int:
    existing_id = connection.execute(
        select(matches.c.id).where(
            matches.c.provider == API_FOOTBALL_PROVIDER,
            matches.c.provider_id == str(fixture.provider_id),
        )
    ).scalar_one_or_none()
    values = {
        "date": fixture.date,
        "home_team_id": team_ids[fixture.home_team_provider_id],
        "away_team_id": team_ids[fixture.away_team_provider_id],
        "home_goals": fixture.home_goals,
        "away_goals": fixture.away_goals,
        "competition_type": "FIFA World Cup",
        "country": "World",
        "neutral_venue": True,
        "provider": API_FOOTBALL_PROVIDER,
        "provider_id": str(fixture.provider_id),
    }
    if existing_id is None:
        result = connection.execute(matches.insert().values(**values))
        return int(result.inserted_primary_key[0])
    connection.execute(update(matches).where(matches.c.id == existing_id).values(**values))
    return int(existing_id)


def sync_api_football_data(
    engine: Engine,
    client: ApiFootballClient,
    league: int = 1,
    season: int = 2026,
    fixture_ids: list[int] | None = None,
    as_of: datetime | None = None,
) -> dict[str, int | str]:
    """Synchronize World Cup squads, injuries and optional fixture lineups/stats."""
    init_database(engine)
    snapshot_date = (as_of or datetime.now(UTC)).date()
    remote_teams = client.world_cup_teams(league=league, season=season)
    remote_players = [
        player
        for team in remote_teams
        for player in client.squad(team.provider_id)
    ]
    remote_injuries = client.injuries(
        league=league,
        season=season,
        as_of=snapshot_date,
    )
    fixture_ids = fixture_ids or []

    with engine.begin() as connection:
        team_ids = _upsert_api_football_teams(connection, remote_teams)
        player_ids = _upsert_api_football_players(
            connection,
            remote_players,
            team_ids,
        )
        injury_player_ids = {
            player_ids[injury.player_provider_id]
            for injury in remote_injuries
            if injury.player_provider_id in player_ids
        }
        for team_id in set(team_ids.values()):
            connection.execute(
                delete(squads).where(
                    squads.c.team_id == team_id,
                    squads.c.as_of_date == snapshot_date,
                )
            )
        squad_rows = [
            {
                "team_id": team_ids[player.team_provider_id],
                "player_id": player_ids[player.provider_id],
                "as_of_date": snapshot_date,
                "available": player_ids[player.provider_id] not in injury_player_ids,
                "attacking_rating": 70.0,
                "defensive_rating": 70.0,
            }
            for player in remote_players
            if player.team_provider_id in team_ids and player.provider_id in player_ids
        ]
        if squad_rows:
            connection.execute(squads.insert(), squad_rows)

        if player_ids:
            connection.execute(
                delete(injuries).where(
                    injuries.c.reported_at == snapshot_date,
                    injuries.c.player_id.in_(list(player_ids.values())),
                )
            )
        injury_rows = [
            {
                "player_id": player_ids[injury.player_provider_id],
                "reported_at": injury.reported_at,
                "status": injury.status,
            }
            for injury in remote_injuries
            if injury.player_provider_id in player_ids
        ]
        if injury_rows:
            connection.execute(injuries.insert(), injury_rows)

        lineup_count = 0
        stat_count = 0
        position_by_provider_id = {
            player.provider_id: player.position for player in remote_players
        }
        for fixture_id in fixture_ids:
            fixture = client.fixture(fixture_id)
            remote_lineup = client.fixture_lineup(fixture_id)
            remote_stats = client.fixture_player_stats(fixture_id)
            match_id = _upsert_api_football_match(connection, fixture, team_ids)
            connection.execute(delete(lineups).where(lineups.c.match_id == match_id))
            connection.execute(
                delete(player_match_stats).where(
                    player_match_stats.c.match_id == match_id
                )
            )
            lineup_rows = [
                {
                    "match_id": match_id,
                    "player_id": player_ids[item.player_provider_id],
                    "team_id": team_ids[item.team_provider_id],
                    "starter": item.starter,
                }
                for item in remote_lineup
                if item.player_provider_id in player_ids
                and item.team_provider_id in team_ids
            ]
            if lineup_rows:
                connection.execute(lineups.insert(), lineup_rows)
                lineup_count += len(lineup_rows)
            stat_rows = [
                {
                    "player_id": player_ids[item.player_provider_id],
                    "match_id": match_id,
                    "minutes_played": item.minutes_played,
                    "goals": item.goals,
                    "assists": item.assists,
                    "rating": item.rating,
                }
                for item in remote_stats
                if item.player_provider_id in player_ids
            ]
            if stat_rows:
                connection.execute(player_match_stats.insert(), stat_rows)
                stat_count += len(stat_rows)
            for item in remote_stats:
                if item.player_provider_id not in player_ids or item.rating is None:
                    continue
                position = position_by_provider_id.get(item.player_provider_id) or ""
                rating = item.rating * 10.0
                values = {}
                if position in {"Attacker", "Midfielder"}:
                    values["attacking_rating"] = rating
                if position in {"Defender", "Goalkeeper", "Midfielder"}:
                    values["defensive_rating"] = rating
                if values:
                    connection.execute(
                        update(squads)
                        .where(
                            squads.c.player_id == player_ids[item.player_provider_id],
                            squads.c.as_of_date == snapshot_date,
                        )
                        .values(**values)
                    )

    return {
        "provider": API_FOOTBALL_PROVIDER,
        "league": league,
        "season": season,
        "teams": len(remote_teams),
        "players": len(player_ids),
        "injuries": len(injury_rows),
        "fixtures": len(fixture_ids),
        "lineups": lineup_count,
        "player_match_stats": stat_count,
        "as_of": snapshot_date.isoformat(),
    }


def load_player_features_from_database(
    engine: Engine,
    as_of_date: datetime | None = None,
) -> pd.DataFrame:
    """Load the latest synced player snapshot in hybrid-model CSV shape."""
    with engine.connect() as connection:
        snapshot_date = (
            as_of_date.date()
            if as_of_date is not None
            else connection.execute(select(func.max(squads.c.as_of_date))).scalar_one()
        )
        if snapshot_date is None:
            return pd.DataFrame(
                columns=[
                    "team",
                    "player",
                    "attacking_rating",
                    "defensive_rating",
                    "available",
                ]
            )
        rows = connection.execute(
            select(
                teams.c.name.label("team"),
                players.c.name.label("player"),
                squads.c.attacking_rating,
                squads.c.defensive_rating,
                squads.c.available,
            )
            .join(squads, squads.c.team_id == teams.c.id)
            .join(players, players.c.id == squads.c.player_id)
            .where(squads.c.as_of_date == snapshot_date)
        ).mappings().all()
    return pd.DataFrame(rows)


def persist_elo_model(engine: Engine, model: EloPoissonModel) -> None:
    if model.parameters is None:
        raise ValueError("Cannot persist an unfitted model")
    with engine.begin() as connection:
        existing_id = connection.execute(
            select(models.c.id).where(models.c.model_version == model.model_version)
        ).scalar_one_or_none()
        values = {
            "model_version": model.model_version,
            "model_type": "elo_poisson",
            "home_advantage_gamma": model.parameters.home_advantage,
            "time_decay_xi": model.parameters.time_decay_xi,
            "competition_weights_json": model.elo_config.competition_k_factors,
            "elo_k_factor": model.elo_config.default_k_factor,
        }
        if existing_id is None:
            connection.execute(models.insert().values(**values))
        else:
            connection.execute(update(models).where(models.c.id == existing_id).values(**values))

        team_ids = _ensure_teams(connection, list(model.ratings))
        connection.execute(
            rating_snapshots.insert(),
            [
                {
                    "computed_at": datetime.now(UTC),
                    "team_id": team_ids[team],
                    "model_version": model.model_version,
                    "elo": rating,
                }
                for team, rating in model.ratings.items()
            ],
        )


def persist_prediction(
    engine: Engine,
    prediction: MatchPrediction,
    model_version: str,
) -> int:
    with engine.begin() as connection:
        team_ids = _ensure_teams(connection, [prediction.home_team, prediction.away_team])
        result = connection.execute(
            predictions.insert().values(
                home_team_id=team_ids[prediction.home_team],
                away_team_id=team_ids[prediction.away_team],
                home_win_prob=prediction.home_win_prob,
                draw_prob=prediction.draw_prob,
                away_win_prob=prediction.away_win_prob,
                most_likely_score=prediction.most_likely_score,
                score_matrix_json=prediction.score_matrix,
                model_version=model_version,
            )
        )
        return int(result.inserted_primary_key[0])


def persist_simulation_results(
    engine: Engine,
    result: pd.DataFrame,
    simulation_run_id: str,
    model_version: str,
) -> int:
    with engine.begin() as connection:
        team_ids = _ensure_teams(connection, result["team"].astype(str).tolist())
        connection.execute(
            simulation_results.insert(),
            [
                {
                    "simulation_run_id": simulation_run_id,
                    "team_id": team_ids[row.team],
                    "group_qualify_prob": float(row.group_qualify_prob),
                    "round_of_32_prob": float(row.round_of_32_prob),
                    "round_of_16_prob": float(row.round_of_16_prob),
                    "quarter_final_prob": float(row.quarter_final_prob),
                    "semi_final_prob": float(row.semi_final_prob),
                    "final_prob": float(row.final_prob),
                    "champion_prob": float(row.champion_prob),
                    "model_version": model_version,
                }
                for row in result.itertuples(index=False)
            ],
        )
    return len(result)


def write_schema_sql(destination: str | Path, database_url: str = "postgresql+psycopg://") -> Path:
    from sqlalchemy.schema import CreateTable

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_mock_engine(database_url, executor=lambda *args, **kwargs: None)
    statements: Iterable[str] = (
        str(CreateTable(table).compile(engine)).strip() + ";"
        for table in metadata.sorted_tables
    )
    path.write_text("\n\n".join(statements) + "\n", encoding="utf-8")
    return path
