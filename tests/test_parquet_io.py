from __future__ import annotations

from pathlib import Path

from worldcup_predictor.store.parquet_io import export_matches_parquet, read_parquet


def test_export_matches_parquet_round_trip(tmp_path: Path) -> None:
    output = tmp_path / "matches.parquet"

    summary = export_matches_parquet(
        "data/examples/synthetic_matches.csv",
        output,
    )
    restored = read_parquet(output)

    assert summary["rows"] == 18
    assert len(restored) == 18
    assert {"date", "home_team", "away_team", "home_goals", "away_goals"} <= set(
        restored.columns
    )

