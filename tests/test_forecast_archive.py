from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from worldcup_predictor.store.forecast_archive import write_forecast_with_history


def test_write_forecast_with_history_writes_live_and_dated_copy(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"team": ["Spain"], "champion_prob": [0.3]})
    output = tmp_path / "simulation.csv"

    archive_path = write_forecast_with_history(
        frame, output, as_of=date(2026, 7, 13)
    )

    assert output.is_file()
    assert archive_path == tmp_path / "forecast_history" / "simulation_2026-07-13.csv"
    assert archive_path.read_text() == output.read_text()

    # Same-day rerun overwrites that day's snapshot instead of duplicating.
    updated = pd.DataFrame({"team": ["Spain"], "champion_prob": [0.4]})
    second_path = write_forecast_with_history(
        updated, output, as_of=date(2026, 7, 13)
    )
    assert second_path == archive_path
    assert "0.4" in archive_path.read_text()
