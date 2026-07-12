from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master/results.csv"
)

# Companion file with penalty-shootout winners for matches that were drawn
# after full time (the results file scores include extra time but not pens).
SHOOTOUTS_URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master/shootouts.csv"
)


def download_international_results(
    destination: str | Path,
    source_url: str = INTERNATIONAL_RESULTS_URL,
) -> dict[str, Any]:
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "worldcup-predictor/0.1"},
    )

    digest = hashlib.sha256()
    byte_count = 0
    temporary_path: Path | None = None
    try:
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            tempfile.NamedTemporaryFile(
                mode="wb",
                dir=output.parent,
                prefix=f".{output.name}.",
                delete=False,
            ) as temporary,
        ):
            temporary_path = Path(temporary.name)
            while chunk := response.read(1024 * 1024):
                temporary.write(chunk)
                digest.update(chunk)
                byte_count += len(chunk)
        temporary_path.replace(output)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    metadata = {
        "source_url": source_url,
        "license": "CC0-1.0",
        "downloaded_at": datetime.now(UTC).isoformat(),
        "sha256": digest.hexdigest(),
        "bytes": byte_count,
        "output": str(output),
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary_metadata, metadata_path)
    return metadata

