from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.evaluation.backtest import time_split_backtest
from worldcup_predictor.compute import (
    ComputeDevice,
    DeviceUnavailableError,
    resolve_device,
)
from worldcup_predictor.evaluation.rating_v2_backtest import backtest_rating_v2_poisson
from worldcup_predictor.ingestion.api_football import ApiFootballClient
from worldcup_predictor.ingestion.download import download_international_results
from worldcup_predictor.ingestion.matches import load_matches
from worldcup_predictor.features.player_features import (
    PlayerAdjustedPredictor,
    aggregate_team_player_features,
    load_players,
)
from worldcup_predictor.models.bayesian import BayesianHierarchicalModel
from worldcup_predictor.models.dixon_coles import DixonColesModel
from worldcup_predictor.models.elo_poisson import EloPoissonModel
from worldcup_predictor.models.rating_v2_poisson import RatingV2PoissonModel
from worldcup_predictor.models.tournament_value import (
    TournamentValueConfig,
    TournamentValueNetwork,
)
from worldcup_predictor.simulation.actual_results import load_knockout_winners_from_files
from worldcup_predictor.simulation.batch_tournament import load_batch_elo_poisson_simulator
from worldcup_predictor.simulation.tournament import TournamentConfig, load_elo_poisson_simulator
from worldcup_predictor.store.db import (
    engine_from_url,
    init_database,
    load_player_features_from_database,
    load_matches_csv,
    sync_api_football_data,
    table_names,
    write_schema_sql,
)
from worldcup_predictor.store.parquet_io import export_matches_parquet
from worldcup_predictor.workflows.catch_up import catch_up
from worldcup_predictor.workflows.dynamic_update import (
    MatchResultInput,
    run_dynamic_update,
)
from worldcup_predictor.workflows.distilled_value import (
    predict_distilled_value_engine,
    train_distilled_value_engine,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="worldcup-predictor",
        description="Train and use the World Football Elo + Poisson model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Fit a model from a match CSV")
    train.add_argument("--matches", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)

    train_dc = subparsers.add_parser("train-dixon-coles", help="Fit full Dixon-Coles attack/defense model")
    train_dc.add_argument("--matches", required=True, type=Path)
    train_dc.add_argument("--output", required=True, type=Path)
    train_dc.add_argument("--since")
    train_dc.add_argument("--max-iterations", type=int, default=5_000)
    train_dc.add_argument("--max-function-evaluations", type=int)

    train_bayes = subparsers.add_parser("train-bayesian", help="Fit empirical-Bayes hierarchical model")
    train_bayes.add_argument("--matches", required=True, type=Path)
    train_bayes.add_argument("--output", required=True, type=Path)
    train_bayes.add_argument("--posterior-draws", type=int, default=200)
    train_bayes.add_argument("--since")
    train_bayes.add_argument("--max-iterations", type=int, default=500)

    train_rating_v2 = subparsers.add_parser(
        "train-rating-v2",
        help="Fit experimental Rating Engine v2 + Poisson model",
    )
    train_rating_v2.add_argument("--matches", required=True, type=Path)
    train_rating_v2.add_argument("--output", required=True, type=Path)

    fetch_data = subparsers.add_parser(
        "fetch-data", help="Download the CC0 international results dataset"
    )
    fetch_data.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/international_results.csv"),
    )

    backtest = subparsers.add_parser(
        "backtest", help="Run a chronological train/test evaluation"
    )
    backtest.add_argument("--matches", required=True, type=Path)
    backtest.add_argument("--cutoff", required=True)
    backtest.add_argument("--calibration-bins", type=int, default=10)
    backtest.add_argument("--predictions-output", type=Path)
    backtest.add_argument("--calibration-output", type=Path)

    backtest_rating_v2 = subparsers.add_parser(
        "backtest-rating-v2",
        help="Run chronological evaluation for Rating Engine v2 + Poisson",
    )
    backtest_rating_v2.add_argument("--matches", required=True, type=Path)
    backtest_rating_v2.add_argument("--cutoff", required=True)
    backtest_rating_v2.add_argument("--calibration-bins", type=int, default=10)
    backtest_rating_v2.add_argument("--predictions-output", type=Path)
    backtest_rating_v2.add_argument("--calibration-output", type=Path)

    parquet = subparsers.add_parser(
        "export-parquet", help="Validate match CSV and save processed Parquet"
    )
    parquet.add_argument("--matches", required=True, type=Path)
    parquet.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/matches.parquet"),
    )

    schema = subparsers.add_parser("write-schema", help="Write SQL schema DDL")
    schema.add_argument(
        "--output",
        type=Path,
        default=Path("docs/schema.sql"),
    )

    init_db = subparsers.add_parser("init-db", help="Create database tables")
    init_db.add_argument("--database-url", required=True)

    load_db = subparsers.add_parser("load-matches-db", help="Load match CSV into database")
    load_db.add_argument("--database-url", required=True)
    load_db.add_argument("--matches", required=True, type=Path)

    rankings = subparsers.add_parser("rankings", help="Show Elo rankings")
    rankings.add_argument("--model", required=True, type=Path)
    rankings.add_argument("--limit", type=int, default=20)

    predict = subparsers.add_parser("predict", help="Predict one match")
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--home", required=True)
    predict.add_argument("--away", required=True)
    venue = predict.add_mutually_exclusive_group()
    venue.add_argument("--neutral", action="store_true", default=True)
    venue.add_argument("--home-venue", action="store_false", dest="neutral")
    predict.add_argument(
        "--include-score-matrix",
        action="store_true",
        help="Include the full 0..N by 0..N score matrix",
    )

    benchmark = subparsers.add_parser(
        "benchmark-prediction",
        help="Measure repeat single-match prediction throughput by device",
    )
    benchmark.add_argument("--model", required=True, type=Path)
    benchmark.add_argument("--home", required=True)
    benchmark.add_argument("--away", required=True)
    benchmark.add_argument("--iterations", type=int, default=10_000)
    benchmark.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        action="append",
        default=[],
        help="Repeat to benchmark several devices; defaults to CPU and CUDA.",
    )

    benchmark_batch = subparsers.add_parser(
        "benchmark-batch-prediction",
        help="Measure batched match prediction throughput by device",
    )
    benchmark_batch.add_argument("--model", required=True, type=Path)
    benchmark_batch.add_argument("--home", required=True)
    benchmark_batch.add_argument("--away", required=True)
    benchmark_batch.add_argument("--batch-size", type=int, default=10_000)
    benchmark_batch.add_argument("--repeat", type=int, default=5)
    benchmark_batch.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        action="append",
        default=[],
        help="Repeat to benchmark several devices; defaults to CPU and CUDA.",
    )
    predict.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Prediction backend; auto uses CUDA when a usable device is available.",
    )

    predict_dc = subparsers.add_parser("predict-dixon-coles", help="Predict with a Dixon-Coles model")
    predict_dc.add_argument("--model", required=True, type=Path)
    predict_dc.add_argument("--home", required=True)
    predict_dc.add_argument("--away", required=True)
    predict_dc.add_argument("--neutral", action="store_true", default=True)

    predict_bayes = subparsers.add_parser("predict-bayesian", help="Predict with a Bayesian model")
    predict_bayes.add_argument("--model", required=True, type=Path)
    predict_bayes.add_argument("--home", required=True)
    predict_bayes.add_argument("--away", required=True)
    predict_bayes.add_argument("--neutral", action="store_true", default=True)

    predict_rating_v2 = subparsers.add_parser(
        "predict-rating-v2",
        help="Predict with the experimental Rating Engine v2 + Poisson model",
    )
    predict_rating_v2.add_argument("--model", required=True, type=Path)
    predict_rating_v2.add_argument("--home", required=True)
    predict_rating_v2.add_argument("--away", required=True)
    predict_rating_v2.add_argument("--neutral", action="store_true", default=True)
    predict_rating_v2.add_argument(
        "--include-score-matrix",
        action="store_true",
    )

    predict_players = subparsers.add_parser("predict-player-adjusted", help="Predict with team + player adjustments")
    predict_players.add_argument("--model", required=True, type=Path)
    predict_players.add_argument("--players", type=Path)
    predict_players.add_argument("--database-url")
    predict_players.add_argument("--home", required=True)
    predict_players.add_argument("--away", required=True)
    predict_players.add_argument("--neutral", action="store_true", default=True)

    simulate = subparsers.add_parser(
        "simulate", help="Run a Monte Carlo World Cup tournament simulation"
    )
    simulate.add_argument(
        "--model",
        type=Path,
        default=Path("models/elo_poisson_current.json"),
        help="Model file; the pre-simulation catch-up refits it in place "
        "unless --offline is given.",
    )
    simulate.add_argument(
        "--groups",
        type=Path,
        default=Path("data/worldcup/groups_2026.csv"),
    )
    simulate.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/worldcup/fixtures_2026.csv"),
    )
    simulate.add_argument(
        "--matches",
        type=Path,
        default=Path("data/raw/international_results.csv"),
        help="Full match history used for the catch-up refit and for pinning "
        "knockout matches already played.",
    )
    simulate.add_argument(
        "--shootouts",
        type=Path,
        default=Path("data/raw/shootouts.csv"),
    )
    simulate.add_argument("--simulations", type=int, default=1000)
    simulate.add_argument("--seed", type=int)
    simulate.add_argument("--output", type=Path)
    simulate.add_argument("--limit", type=int, default=20)
    simulate.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Score-matrix backend; auto uses CUDA when a usable device is available.",
    )
    simulate.add_argument(
        "--offline",
        action="store_true",
        help="Skip the download-and-refit catch-up and use existing files as-is.",
    )
    simulate.add_argument(
        "--no-condition-knockouts",
        action="store_true",
        help="Do not pin knockout matches already played to their real winners.",
    )

    catch_up_parser = subparsers.add_parser(
        "catch-up",
        help="Download the latest results, fill fixture scores and refit the model",
    )
    catch_up_parser.add_argument(
        "--matches",
        type=Path,
        default=Path("data/raw/international_results.csv"),
    )
    catch_up_parser.add_argument(
        "--shootouts",
        type=Path,
        default=Path("data/raw/shootouts.csv"),
    )
    catch_up_parser.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/worldcup/fixtures_2026.csv"),
    )
    catch_up_parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/elo_poisson_current.json"),
    )
    catch_up_parser.add_argument(
        "--offline",
        action="store_true",
        help="Fill fixtures and refit from existing local files without downloading.",
    )

    benchmark_sim_b = subparsers.add_parser(
        "benchmark-simulation-b",
        help="Compare the stable simulator with the experimental batch simulator",
    )
    benchmark_sim_b.add_argument("--model", required=True, type=Path)
    benchmark_sim_b.add_argument(
        "--groups",
        type=Path,
        default=Path("data/worldcup/groups_2026.csv"),
    )
    benchmark_sim_b.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/worldcup/fixtures_2026.csv"),
    )
    benchmark_sim_b.add_argument("--simulations", type=int, default=1000)
    benchmark_sim_b.add_argument("--seed", type=int)
    benchmark_sim_b.add_argument(
        "--standard-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    benchmark_sim_b.add_argument(
        "--batch-device",
        choices=("auto", "cpu", "cuda"),
        default="cuda",
    )

    benchmark_value_d = subparsers.add_parser(
        "benchmark-value-network-d",
        help="Train and evaluate the experimental tournament value network",
    )
    benchmark_value_d.add_argument("--model", required=True, type=Path)
    benchmark_value_d.add_argument(
        "--groups",
        type=Path,
        default=Path("data/worldcup/groups_2026.csv"),
    )
    benchmark_value_d.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/worldcup/fixtures_2026.csv"),
    )
    benchmark_value_d.add_argument("--label-simulations", type=int, default=2000)
    benchmark_value_d.add_argument("--seed", type=int)
    benchmark_value_d.add_argument("--epochs", type=int, default=500)
    benchmark_value_d.add_argument("--hidden-units", type=int, default=32)
    benchmark_value_d.add_argument("--learning-rate", type=float, default=0.01)
    benchmark_value_d.add_argument(
        "--label-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    benchmark_value_d.add_argument(
        "--train-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    benchmark_value_d.add_argument(
        "--predict-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )

    train_value_bd = subparsers.add_parser(
        "train-value-engine-bd",
        help="Train the B+D distilled tournament value engine",
    )
    train_value_bd.add_argument("--model", required=True, type=Path)
    train_value_bd.add_argument(
        "--groups",
        type=Path,
        default=Path("data/worldcup/groups_2026.csv"),
    )
    train_value_bd.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/worldcup/fixtures_2026.csv"),
    )
    train_value_bd.add_argument(
        "--output",
        type=Path,
        default=Path("models/value_engine_bd_current.json"),
    )
    train_value_bd.add_argument("--target-output", type=Path)
    train_value_bd.add_argument("--prediction-output", type=Path)
    train_value_bd.add_argument("--label-simulations", type=int, default=2000)
    train_value_bd.add_argument("--seed", type=int)
    train_value_bd.add_argument("--epochs", type=int, default=800)
    train_value_bd.add_argument("--hidden-units", type=int, default=32)
    train_value_bd.add_argument("--learning-rate", type=float, default=0.01)
    train_value_bd.add_argument(
        "--label-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    train_value_bd.add_argument(
        "--train-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    train_value_bd.add_argument(
        "--predict-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )

    predict_value_bd = subparsers.add_parser(
        "predict-value-engine-bd",
        help="Predict champion probabilities with a trained B+D value engine",
    )
    predict_value_bd.add_argument("--model", required=True, type=Path)
    predict_value_bd.add_argument("--value-model", required=True, type=Path)
    predict_value_bd.add_argument(
        "--groups",
        type=Path,
        default=Path("data/worldcup/groups_2026.csv"),
    )
    predict_value_bd.add_argument(
        "--fixtures",
        type=Path,
        default=Path("data/worldcup/fixtures_2026.csv"),
    )
    predict_value_bd.add_argument("--output", type=Path)
    predict_value_bd.add_argument("--limit", type=int, default=20)
    predict_value_bd.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )

    dynamic = subparsers.add_parser("dynamic-update", help="Append a result, refit, repredict and resimulate")
    dynamic.add_argument("--matches", required=True, type=Path)
    dynamic.add_argument("--date", required=True)
    dynamic.add_argument("--home", required=True)
    dynamic.add_argument("--away", required=True)
    dynamic.add_argument("--home-goals", required=True, type=int)
    dynamic.add_argument("--away-goals", required=True, type=int)
    dynamic.add_argument("--competition-type", default="FIFA World Cup")
    dynamic.add_argument("--neutral", action="store_true", default=True)
    dynamic.add_argument("--model-output", type=Path, default=Path("models/elo_poisson_current.json"))
    dynamic.add_argument("--simulation-output", type=Path, default=Path("data/processed/simulation_2026.csv"))
    dynamic.add_argument("--predictions-output", type=Path, default=Path("data/processed/remaining_predictions.csv"))
    dynamic.add_argument("--simulations", type=int, default=1000)

    sync_players = subparsers.add_parser(
        "sync-api-football",
        help="Sync World Cup squads, injuries and optional fixture player data",
    )
    sync_players.add_argument("--database-url", required=True)
    sync_players.add_argument("--api-key")
    sync_players.add_argument("--league", type=int, default=1)
    sync_players.add_argument("--season", type=int, default=2026)
    sync_players.add_argument("--fixture-id", type=int, action="append", default=[])
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    if args.command == "fetch-data":
        metadata = download_international_results(args.output)
        print(json.dumps(metadata, indent=2))
        return

    if args.command == "sync-api-football":
        with ApiFootballClient.from_environment(args.api_key) as client:
            summary = sync_api_football_data(
                engine_from_url(args.database_url),
                client,
                league=args.league,
                season=args.season,
                fixture_ids=args.fixture_id,
            )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train":
        matches = load_matches(args.matches, completed_only=True)
        model = EloPoissonModel().fit(matches)
        model.save(args.output)
        summary = {
            "model_version": model.model_version,
            "matches": len(matches),
            "excluded_unplayed": matches.attrs.get("dropped_unplayed", 0),
            "teams": len(model.ratings),
            "trained_through": model.trained_through,
            "output": str(args.output),
        }
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train-dixon-coles":
        matches = load_matches(args.matches, completed_only=True)
        if args.since:
            matches = matches.loc[matches["date"] >= pd.Timestamp(args.since)].reset_index(drop=True)
        model = DixonColesModel(
            max_iterations=args.max_iterations,
            max_function_evaluations=args.max_function_evaluations,
        ).fit(matches)
        model.save(args.output)
        print(json.dumps({
            "model_version": model.model_version,
            "matches": len(matches),
            "teams": len(model.teams),
            "trained_through": model.trained_through,
            "output": str(args.output),
            "rho": model.parameters.rho if model.parameters else None,
            "optimization_success": model.optimization_success,
            "optimization_message": model.optimization_message,
            "optimization_iterations": model.optimization_iterations,
            "optimization_function_evaluations": model.optimization_function_evaluations,
        }, indent=2))
        return

    if args.command == "train-bayesian":
        matches = load_matches(args.matches, completed_only=True)
        if args.since:
            matches = matches.loc[matches["date"] >= pd.Timestamp(args.since)].reset_index(drop=True)
        model = BayesianHierarchicalModel(
            posterior_draws=args.posterior_draws,
            max_iterations=args.max_iterations,
        ).fit(matches)
        model.save(args.output)
        print(json.dumps({
            "model_version": model.model_version,
            "matches": len(matches),
            "teams": len(model.teams),
            "trained_through": model.trained_through,
            "output": str(args.output),
            "optimization_success": model.optimization_success,
            "optimization_message": model.optimization_message,
        }, indent=2))
        return

    if args.command == "train-rating-v2":
        matches = load_matches(args.matches, completed_only=True)
        model = RatingV2PoissonModel().fit(matches)
        model.save(args.output)
        print(json.dumps({
            "model_version": model.model_version,
            "matches": len(matches),
            "excluded_unplayed": matches.attrs.get("dropped_unplayed", 0),
            "teams": len(model.ratings),
            "trained_through": model.trained_through,
            "output": str(args.output),
        }, indent=2))
        return

    if args.command == "backtest":
        matches = load_matches(args.matches, completed_only=True)
        result = time_split_backtest(
            matches,
            cutoff=args.cutoff,
            calibration_bins=args.calibration_bins,
        )
        if args.predictions_output:
            args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
            result.predictions.to_csv(args.predictions_output, index=False)
        if args.calibration_output:
            args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
            result.calibration.to_csv(args.calibration_output, index=False)
        summary = result.summary()
        summary["excluded_unplayed"] = matches.attrs.get("dropped_unplayed", 0)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "backtest-rating-v2":
        matches = load_matches(args.matches, completed_only=True)
        result = backtest_rating_v2_poisson(
            matches,
            cutoff=args.cutoff,
            calibration_bins=args.calibration_bins,
        )
        if args.predictions_output:
            args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
            result.predictions.to_csv(args.predictions_output, index=False)
        if args.calibration_output:
            args.calibration_output.parent.mkdir(parents=True, exist_ok=True)
            result.calibration.to_csv(args.calibration_output, index=False)
        summary = result.summary()
        summary["excluded_unplayed"] = matches.attrs.get("dropped_unplayed", 0)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "export-parquet":
        summary = export_matches_parquet(args.matches, args.output)
        print(json.dumps(summary, indent=2))
        return

    if args.command == "write-schema":
        output = write_schema_sql(args.output)
        print(json.dumps({"output": str(output), "tables": table_names()}, indent=2))
        return

    if args.command == "init-db":
        init_database(engine_from_url(args.database_url))
        print(json.dumps({"database_url": args.database_url, "tables": table_names()}, indent=2))
        return

    if args.command == "load-matches-db":
        summary = load_matches_csv(
            engine_from_url(args.database_url),
            args.matches,
            completed_only=True,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "dynamic-update":
        summary = run_dynamic_update(
            matches_path=args.matches,
            result=MatchResultInput(
                date=args.date,
                home_team=args.home,
                away_team=args.away,
                home_goals=args.home_goals,
                away_goals=args.away_goals,
                competition_type=args.competition_type,
                neutral_venue=args.neutral,
            ),
            model_output=args.model_output,
            simulation_output=args.simulation_output,
            predictions_output=args.predictions_output,
            simulations=args.simulations,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.command == "predict-dixon-coles":
        prediction = DixonColesModel.load(args.model).predict(args.home, args.away, args.neutral)
        print(json.dumps(prediction.to_dict(include_score_matrix=False), indent=2))
        return

    if args.command == "predict-bayesian":
        model = BayesianHierarchicalModel.load(args.model)
        prediction = model.predict(args.home, args.away, args.neutral)
        interval = model.prediction_interval(args.home, args.away, args.neutral, draws=100)
        payload = prediction.to_dict(include_score_matrix=False)
        payload["prediction_interval_90"] = interval.__dict__
        print(json.dumps(payload, indent=2))
        return

    if args.command == "predict-rating-v2":
        prediction = RatingV2PoissonModel.load(args.model).predict(
            args.home,
            args.away,
            args.neutral,
        )
        print(json.dumps(
            prediction.to_dict(include_score_matrix=args.include_score_matrix),
            indent=2,
        ))
        return

    if args.command == "predict-player-adjusted":
        if bool(args.players) == bool(args.database_url):
            raise ValueError(
                "Specify exactly one of --players or --database-url "
                "for player-adjusted prediction"
            )
        base = EloPoissonModel.load(args.model)
        player_frame = (
            load_players(args.players)
            if args.players
            else load_player_features_from_database(
                engine_from_url(args.database_url)
            )
        )
        adjustments = aggregate_team_player_features(player_frame)
        prediction = PlayerAdjustedPredictor(base, adjustments).predict(args.home, args.away, args.neutral)
        print(json.dumps(prediction.to_dict(include_score_matrix=False), indent=2))
        return

    if args.command == "benchmark-prediction":
        if args.iterations <= 0:
            raise ValueError("iterations must be positive")
        model = EloPoissonModel.load(args.model)
        rows = []
        for device in args.device or ["cpu", "cuda"]:
            try:
                resolved = resolve_device(device)
            except DeviceUnavailableError as exc:
                rows.append({"device": device, "status": "unavailable", "reason": str(exc)})
                continue
            model.predict(args.home, args.away, device=resolved.name)
            started = time.perf_counter()
            for _ in range(args.iterations):
                model.predict(args.home, args.away, device=resolved.name)
            elapsed = time.perf_counter() - started
            rows.append(
                {
                    "device": resolved.name,
                    "status": "ok",
                    "iterations": args.iterations,
                    "seconds": elapsed,
                    "predictions_per_second": args.iterations / elapsed,
                }
            )
        print(json.dumps(rows, indent=2))
        return

    if args.command == "benchmark-batch-prediction":
        if args.batch_size <= 0:
            raise ValueError("batch-size must be positive")
        if args.repeat <= 0:
            raise ValueError("repeat must be positive")
        model = EloPoissonModel.load(args.model)
        batch = [(args.home, args.away, True)] * args.batch_size
        rows = []
        for device in args.device or ["cpu", "cuda"]:
            try:
                resolved = resolve_device(device)
            except DeviceUnavailableError as exc:
                rows.append({"device": device, "status": "unavailable", "reason": str(exc)})
                continue
            model.predict_many(batch, device=resolved.name)
            started = time.perf_counter()
            for _ in range(args.repeat):
                model.predict_many(batch, device=resolved.name)
            elapsed = time.perf_counter() - started
            predictions = args.batch_size * args.repeat
            rows.append(
                {
                    "device": resolved.name,
                    "status": "ok",
                    "batch_size": args.batch_size,
                    "repeat": args.repeat,
                    "predictions": predictions,
                    "seconds": elapsed,
                    "predictions_per_second": predictions / elapsed,
                }
            )
        print(json.dumps(rows, indent=2))
        return

    if args.command == "catch-up":
        summary = catch_up(
            raw_path=args.matches,
            fixtures_path=args.fixtures,
            model_output=args.model_output,
            shootouts_path=args.shootouts,
            offline=args.offline,
        )
        print(json.dumps(summary.to_dict(), indent=2))
        return

    if args.command == "simulate":
        if not args.offline:
            summary = catch_up(
                raw_path=args.matches,
                fixtures_path=args.fixtures,
                model_output=args.model,
                shootouts_path=args.shootouts,
            )
            print(json.dumps({"catch_up": summary.to_dict()}, indent=2))
        knockout_winners = None
        if not args.no_condition_knockouts and args.matches.is_file():
            knockout_winners = load_knockout_winners_from_files(
                raw_path=args.matches,
                config=TournamentConfig.from_csv(args.groups, args.fixtures),
                shootouts_path=args.shootouts,
            )
        simulator = load_elo_poisson_simulator(
            model_path=args.model,
            groups_path=args.groups,
            fixtures_path=args.fixtures,
            random_seed=args.seed,
            device=args.device,
            knockout_winners=knockout_winners,
        )
        result = simulator.run(simulations=args.simulations)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(args.output, index=False)
        print(
            json.dumps(
                {
                    "device": resolve_device(args.device).name,
                    "pinned_knockout_results": len(knockout_winners or {}),
                },
                indent=2,
            )
        )
        print(result.head(args.limit).to_string(index=False))
        return

    model = EloPoissonModel.load(args.model)
    if args.command == "rankings":
        for position, (team, rating) in enumerate(
            model.rankings()[: args.limit], start=1
        ):
            print(f"{position:>3}  {team:<30} {rating:8.2f}")
        return

    if args.command == "benchmark-simulation-b":
        if args.simulations <= 0:
            raise ValueError("simulations must be positive")
        rows = []
        standard = load_elo_poisson_simulator(
            model_path=args.model,
            groups_path=args.groups,
            fixtures_path=args.fixtures,
            random_seed=args.seed,
            device=args.standard_device,
        )
        started = time.perf_counter()
        standard_result = standard.run(simulations=args.simulations)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "method": "standard",
                "device": resolve_device(args.standard_device).name,
                "simulations": args.simulations,
                "seconds": elapsed,
                "simulations_per_second": args.simulations / elapsed,
                "top_champion": standard_result.iloc[0]["team"],
                "top_champion_prob": float(standard_result.iloc[0]["champion_prob"]),
            }
        )

        batch = load_batch_elo_poisson_simulator(
            model_path=args.model,
            groups_path=args.groups,
            fixtures_path=args.fixtures,
            random_seed=args.seed,
            device=args.batch_device,
        )
        started = time.perf_counter()
        batch_result = batch.run(simulations=args.simulations)
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "method": "batch_experimental",
                "device": resolve_device(args.batch_device).name,
                "simulations": args.simulations,
                "seconds": elapsed,
                "simulations_per_second": args.simulations / elapsed,
                "top_champion": batch_result.iloc[0]["team"],
                "top_champion_prob": float(batch_result.iloc[0]["champion_prob"]),
            }
        )
        print(json.dumps(rows, indent=2))
        return

    if args.command == "benchmark-value-network-d":
        if args.label_simulations <= 0:
            raise ValueError("label-simulations must be positive")
        if args.epochs <= 0:
            raise ValueError("epochs must be positive")
        base_model = EloPoissonModel.load(args.model)
        tournament_config = TournamentConfig.from_csv(args.groups, args.fixtures)

        label_simulator = load_batch_elo_poisson_simulator(
            model_path=args.model,
            groups_path=args.groups,
            fixtures_path=args.fixtures,
            random_seed=args.seed,
            device=args.label_device,
        )
        started = time.perf_counter()
        target = label_simulator.run(simulations=args.label_simulations)
        label_seconds = time.perf_counter() - started

        value_model = TournamentValueNetwork(
            TournamentValueConfig(
                hidden_units=args.hidden_units,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed if args.seed is not None else 42,
            )
        )
        started = time.perf_counter()
        value_model.fit(
            ratings=base_model.ratings,
            tournament_config=tournament_config,
            target_probabilities=target[["team", "champion_prob"]],
            device=args.train_device,
        )
        train_seconds = time.perf_counter() - started

        started = time.perf_counter()
        prediction = value_model.predict_champion_probabilities(
            ratings=base_model.ratings,
            tournament_config=tournament_config,
            device=args.predict_device,
        )
        predict_seconds = time.perf_counter() - started

        merged = prediction.merge(
            target[["team", "champion_prob"]],
            on="team",
            suffixes=("_value", "_target"),
        )
        error = (
            merged["champion_prob_value"] - merged["champion_prob_target"]
        ).to_numpy(dtype=float)
        payload = {
            "method": "value_network_d",
            "label_simulations": args.label_simulations,
            "label_device": resolve_device(args.label_device).name,
            "train_device": resolve_device(args.train_device).name,
            "predict_device": resolve_device(args.predict_device).name,
            "label_seconds": label_seconds,
            "train_seconds": train_seconds,
            "predict_seconds": predict_seconds,
            "mae": float(np.mean(np.abs(error))),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "max_abs_error": float(np.max(np.abs(error))),
            "target_top_champion": target.iloc[0]["team"],
            "target_top_champion_prob": float(target.iloc[0]["champion_prob"]),
            "value_top_champion": prediction.iloc[0]["team"],
            "value_top_champion_prob": float(prediction.iloc[0]["champion_prob"]),
        }
        print(json.dumps(payload, indent=2))
        return

    if args.command == "train-value-engine-bd":
        if args.label_simulations <= 0:
            raise ValueError("label-simulations must be positive")
        if args.epochs <= 0:
            raise ValueError("epochs must be positive")
        _, _, summary = train_distilled_value_engine(
            model_path=args.model,
            groups_path=args.groups,
            fixtures_path=args.fixtures,
            value_model_output=args.output,
            target_output=args.target_output,
            prediction_output=args.prediction_output,
            label_simulations=args.label_simulations,
            seed=args.seed,
            value_config=TournamentValueConfig(
                hidden_units=args.hidden_units,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                seed=args.seed if args.seed is not None else 42,
            ),
            label_device=args.label_device,
            train_device=args.train_device,
            predict_device=args.predict_device,
        )
        print(json.dumps(summary.to_dict(), indent=2))
        return

    if args.command == "predict-value-engine-bd":
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        prediction = predict_distilled_value_engine(
            model_path=args.model,
            value_model_path=args.value_model,
            groups_path=args.groups,
            fixtures_path=args.fixtures,
            device=args.device,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            prediction.to_csv(args.output, index=False)
        payload = {
            "device": resolve_device(args.device).name,
            "rows": prediction.head(args.limit).to_dict(orient="records"),
            "output": str(args.output) if args.output else None,
        }
        print(json.dumps(payload, indent=2))
        return

    prediction = model.predict(
        home_team=args.home,
        away_team=args.away,
        neutral_venue=args.neutral,
        device=args.device,
    )
    payload = prediction.to_dict(include_score_matrix=args.include_score_matrix)
    payload["device"] = resolve_device(args.device).name
    print(
        json.dumps(
            payload,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
