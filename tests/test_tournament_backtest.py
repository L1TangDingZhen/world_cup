from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from worldcup_predictor.evaluation.tournament_backtest import (
    load_actual_progress,
    stage_probability_table,
)
from worldcup_predictor.simulation.formats import WC32


def test_load_actual_progress_2018_indicators() -> None:
    actual = load_actual_progress("data/worldcup/actual_2018.csv", WC32)

    assert len(actual) == 32
    by_team = actual.set_index("team")
    # France won: every stage reached.
    assert by_team.loc["France"].tolist() == [1] * 6
    # Croatia lost the final.
    croatia = by_team.loc["Croatia"]
    assert croatia["final"] == 1 and croatia["champion"] == 0
    # Germany went out in the group stage.
    assert by_team.loc["Germany"].sum() == 0
    # Reach counts per stage match the bracket structure.
    for stage, count in WC32.teams_reaching.items():
        assert by_team[stage].sum() == count, stage


def test_stage_probability_table_perfect_and_baseline_predictions() -> None:
    actual = load_actual_progress("data/worldcup/actual_2018.csv", WC32)

    # A perfect forecast puts probability 1 on exactly what happened.
    perfect = pd.DataFrame({"team": actual["team"]})
    for stage in WC32.stage_columns:
        perfect[f"{stage}_prob"] = actual[stage].astype(float)
    stage_table, team_table = stage_probability_table(perfect, actual, WC32)
    assert np.allclose(stage_table["brier"], 0.0)
    assert np.allclose(stage_table["skill"], 1.0)
    assert len(team_table) == 32

    # The structural baseline itself must score exactly zero skill.
    baseline = pd.DataFrame({"team": actual["team"]})
    for stage, count in WC32.teams_reaching.items():
        baseline[f"{stage}_prob"] = count / 32.0
    stage_table, _ = stage_probability_table(baseline, actual, WC32)
    assert np.allclose(stage_table["skill"], 0.0)
    assert np.allclose(stage_table["predicted_sum"], stage_table["actual_count"])


def test_stage_probability_table_rejects_team_mismatch() -> None:
    actual = load_actual_progress("data/worldcup/actual_2018.csv", WC32)
    simulation = pd.DataFrame({"team": ["France"]})
    for stage in WC32.stage_columns:
        simulation[f"{stage}_prob"] = 1.0

    with pytest.raises(ValueError, match="disagree on teams"):
        stage_probability_table(simulation, actual, WC32)
