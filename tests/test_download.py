from __future__ import annotations

from pathlib import Path

from worldcup_predictor.ingestion.download import download_international_results


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.position = 0

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int) -> bytes:
        chunk = self.content[self.position : self.position + size]
        self.position += len(chunk)
        return chunk


def test_download_writes_data_and_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    content = b"date,home_team,away_team\\n2025-01-01,A,B\\n"
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse(content),
    )
    output = tmp_path / "results.csv"

    metadata = download_international_results(
        output,
        source_url="https://example.test/results.csv",
    )

    assert output.read_bytes() == content
    assert metadata["bytes"] == len(content)
    assert len(metadata["sha256"]) == 64
    assert output.with_suffix(".csv.metadata.json").is_file()

