"""Dated archiving for tournament forecasts.

Forecast outputs like simulation_2026.csv used to be overwritten on every
run, losing the history of what the model believed at each point of the
tournament. This helper writes the live output and a dated copy under a
forecast_history directory, so the evolution of the forecast can be
reviewed after the fact (and compared against market odds).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def write_forecast_with_history(
    frame: pd.DataFrame,
    output_path: str | Path,
    history_dir: str | Path | None = None,
    as_of: date | None = None,
) -> Path:
    """Write the forecast CSV plus a dated archive copy; return the copy path.

    Re-running on the same day overwrites that day's archive entry, so the
    history keeps one snapshot per output file per day.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    stamp = (as_of or date.today()).isoformat()
    history = (
        Path(history_dir) if history_dir is not None
        else output.parent / "forecast_history"
    )
    history.mkdir(parents=True, exist_ok=True)
    archive_path = history / f"{output.stem}_{stamp}{output.suffix}"
    frame.to_csv(archive_path, index=False)
    return archive_path
