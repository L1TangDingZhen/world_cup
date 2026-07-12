from __future__ import annotations

from pathlib import Path

import pandas as pd

from worldcup_predictor.ingestion.matches import load_matches


def write_parquet(frame: pd.DataFrame, destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def export_matches_parquet(
    source_csv: str | Path,
    destination: str | Path,
    completed_only: bool = True,
) -> dict[str, object]:
    matches = load_matches(source_csv, completed_only=completed_only)
    path = write_parquet(matches, destination)
    return {
        "source": str(source_csv),
        "output": str(path),
        "rows": len(matches),
        "excluded_unplayed": matches.attrs.get("dropped_unplayed", 0),
        "date_min": matches["date"].min().date().isoformat(),
        "date_max": matches["date"].max().date().isoformat(),
    }

