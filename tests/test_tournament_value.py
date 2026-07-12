from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_predictor.models.tournament_value import (
    TournamentValueConfig,
    TournamentValueNetwork,
)
from worldcup_predictor.simulation.tournament import TournamentConfig


def test_tournament_value_network_fits_champion_distribution() -> None:
    tournament_config = TournamentConfig.from_csv(
        "data/worldcup/groups_2026.csv",
        "data/worldcup/fixtures_2026.csv",
    )
    teams = tournament_config.groups["team"].tolist()
    ratings = {team: 1500.0 for team in teams}
    ratings["Spain"] = 1900.0
    ratings["Argentina"] = 1850.0
    ratings["France"] = 1830.0
    target = pd.DataFrame(
        {
            "team": teams,
            "champion_prob": np.full(len(teams), 0.2 / (len(teams) - 3)),
        }
    )
    target.loc[target["team"].eq("Spain"), "champion_prob"] = 0.35
    target.loc[target["team"].eq("Argentina"), "champion_prob"] = 0.25
    target.loc[target["team"].eq("France"), "champion_prob"] = 0.20

    model = TournamentValueNetwork(
        TournamentValueConfig(epochs=80, hidden_units=16, learning_rate=0.02)
    ).fit(ratings, tournament_config, target, device="cpu")
    prediction = model.predict_champion_probabilities(
        ratings,
        tournament_config,
        device="cpu",
    )

    assert prediction["champion_prob"].sum() == pytest.approx(1.0)
    assert prediction.iloc[0]["team"] in {"Spain", "Argentina", "France"}
    merged = prediction.merge(target, on="team", suffixes=("_pred", "_target"))
    assert np.mean(
        np.abs(merged["champion_prob_pred"] - merged["champion_prob_target"])
    ) < 0.04


def test_tournament_value_network_round_trip(tmp_path) -> None:
    tournament_config = TournamentConfig.from_csv(
        "data/worldcup/groups_2026.csv",
        "data/worldcup/fixtures_2026.csv",
    )
    teams = tournament_config.groups["team"].tolist()
    ratings = {team: 1500.0 for team in teams}
    ratings["Spain"] = 1900.0
    target = pd.DataFrame(
        {
            "team": teams,
            "champion_prob": np.full(len(teams), 0.5 / (len(teams) - 1)),
        }
    )
    target.loc[target["team"].eq("Spain"), "champion_prob"] = 0.5
    model = TournamentValueNetwork(
        TournamentValueConfig(epochs=20, hidden_units=8, learning_rate=0.02)
    ).fit(ratings, tournament_config, target, device="cpu")

    destination = tmp_path / "value.json"
    model.save(destination)
    restored = TournamentValueNetwork.load(destination)

    original = model.predict_champion_probabilities(ratings, tournament_config, device="cpu")
    loaded = restored.predict_champion_probabilities(ratings, tournament_config, device="cpu")
    merged = original.merge(loaded, on="team", suffixes=("_original", "_loaded"))

    assert np.allclose(
        merged["champion_prob_original"],
        merged["champion_prob_loaded"],
    )
