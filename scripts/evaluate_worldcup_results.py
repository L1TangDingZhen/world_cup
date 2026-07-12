from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from worldcup_predictor.evaluation.metrics import (
    brier_score,
    log_loss,
    ranked_probability_score,
)
from worldcup_predictor.models import EloPoissonModel


OUTCOME_LABELS = ["home_win", "draw", "away_win"]


def outcome_index(home_goals: int, away_goals: int) -> int:
    if home_goals > away_goals:
        return 0
    if home_goals == away_goals:
        return 1
    return 2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--model", default="models/elo_poisson_current.json")
    parser.add_argument("--from-date", default="2026-06-13")
    parser.add_argument(
        "--output",
        default="data/processed/post_june12_worldcup_comparison.csv",
    )
    parser.add_argument(
        "--dynamic-update",
        action="store_true",
        help="After scoring each match, update Elo ratings before the next prediction.",
    )
    args = parser.parse_args()

    model = EloPoissonModel.load(args.model)
    raw = pd.read_csv(args.results)
    actual = raw[
        (raw["tournament"] == "FIFA World Cup")
        & (raw["date"] >= args.from_date)
        & raw["home_score"].notna()
        & raw["away_score"].notna()
    ].copy()

    rows: list[dict[str, object]] = []
    probabilities: list[list[float]] = []
    outcomes: list[int] = []
    for match in actual.itertuples(index=False):
        prediction = model.predict(
            match.home_team,
            match.away_team,
            neutral_venue=bool(match.neutral),
        )
        home_goals = int(match.home_score)
        away_goals = int(match.away_score)
        actual_outcome_index = outcome_index(home_goals, away_goals)
        p = [
            prediction.home_win_prob,
            prediction.draw_prob,
            prediction.away_win_prob,
        ]
        predicted_outcome_index = int(np.argmax(p))
        matrix = np.asarray(prediction.score_matrix)
        actual_score_probability = (
            float(matrix[home_goals, away_goals])
            if home_goals < matrix.shape[0] and away_goals < matrix.shape[1]
            else 0.0
        )
        actual_score = f"{home_goals}-{away_goals}"
        rows.append(
            {
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "actual_score": actual_score,
                "actual_outcome": OUTCOME_LABELS[actual_outcome_index],
                "predicted_outcome": OUTCOME_LABELS[predicted_outcome_index],
                "outcome_hit": actual_outcome_index == predicted_outcome_index,
                "home_win_prob": prediction.home_win_prob,
                "draw_prob": prediction.draw_prob,
                "away_win_prob": prediction.away_win_prob,
                "actual_outcome_prob": p[actual_outcome_index],
                "most_likely_score": prediction.most_likely_score,
                "score_hit": prediction.most_likely_score == actual_score,
                "actual_score_prob": actual_score_probability,
                "expected_home_goals": prediction.expected_home_goals,
                "expected_away_goals": prediction.expected_away_goals,
            }
        )
        probabilities.append(p)
        outcomes.append(actual_outcome_index)
        if args.dynamic_update:
            model.update_ratings(
                home_team=match.home_team,
                away_team=match.away_team,
                home_goals=home_goals,
                away_goals=away_goals,
                competition_type=match.tournament,
                neutral_venue=bool(match.neutral),
            )

    result = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)

    probability_array = np.asarray(probabilities)
    outcome_array = np.asarray(outcomes)
    summary = pd.Series(
        {
            "matches": len(result),
            "outcome_accuracy": float(result["outcome_hit"].mean()),
            "exact_score_accuracy": float(result["score_hit"].mean()),
            "mean_actual_outcome_prob": float(result["actual_outcome_prob"].mean()),
            "rps": ranked_probability_score(probability_array, outcome_array),
            "log_loss": log_loss(probability_array, outcome_array),
            "brier_score": brier_score(probability_array, outcome_array),
            "home_wins_actual": int((result["actual_outcome"] == "home_win").sum()),
            "draws_actual": int((result["actual_outcome"] == "draw").sum()),
            "away_wins_actual": int((result["actual_outcome"] == "away_win").sum()),
            "home_wins_predicted": int(
                (result["predicted_outcome"] == "home_win").sum()
            ),
            "draws_predicted": int((result["predicted_outcome"] == "draw").sum()),
            "away_wins_predicted": int(
                (result["predicted_outcome"] == "away_win").sum()
            ),
            "output": str(output),
        }
    )
    print(summary.to_string())
    print("\nBy date:")
    print(
        result.groupby("date")
        .agg(
            matches=("date", "size"),
            outcome_hits=("outcome_hit", "sum"),
            score_hits=("score_hit", "sum"),
            mean_actual_prob=("actual_outcome_prob", "mean"),
        )
        .to_string()
    )
    print("\nRows:")
    columns = [
        "date",
        "home_team",
        "away_team",
        "actual_score",
        "actual_outcome",
        "predicted_outcome",
        "outcome_hit",
        "actual_outcome_prob",
        "most_likely_score",
        "score_hit",
    ]
    print(
        result[columns].to_string(
            index=False,
            formatters={"actual_outcome_prob": "{:.3f}".format},
        )
    )


if __name__ == "__main__":
    main()
