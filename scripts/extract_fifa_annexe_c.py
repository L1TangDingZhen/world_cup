from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
from pathlib import Path

import pandas as pd


SOURCE_URL = "https://digitalhub.fifa.com/m/636f5c9c6f29771f/original/FWC2026_regulations_EN.pdf"
SLOT_COLUMNS = ["T_M79", "T_M85", "T_M81", "T_M74", "T_M82", "T_M77", "T_M87", "T_M80"]
PATTERN = re.compile(r"^\s*(\d{1,3})\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s+(3[A-L])\s*$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/worldcup/third_place_mapping_2026.csv"),
    )
    args = parser.parse_args()

    text = subprocess.check_output(
        ["pdftotext", "-layout", "-f", "80", "-l", "97", str(args.pdf), "-"],
        text=True,
    )
    rows = []
    for line in text.splitlines():
        match = PATTERN.match(line)
        if not match:
            continue
        option = int(match.group(1))
        groups = [value[1] for value in match.groups()[1:]]
        rows.append(
            {
                "option": option,
                "qualifying_groups": "".join(sorted(groups)),
                **dict(zip(SLOT_COLUMNS, groups, strict=True)),
            }
        )

    if len(rows) != 495 or {row["option"] for row in rows} != set(range(1, 496)):
        raise RuntimeError(f"Expected 495 Annex C options, found {len(rows)}")
    if len({row["qualifying_groups"] for row in rows}) != 495:
        raise RuntimeError("Annex C has duplicate qualifying-group combinations")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("option").to_csv(args.output, index=False)
    metadata = args.output.with_suffix(".metadata.txt")
    metadata.write_text(
        "source_url=" + SOURCE_URL + "\n"
        "source_sha256=" + hashlib.sha256(args.pdf.read_bytes()).hexdigest() + "\n"
        "source_pages=80-97\n"
        "options=495\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

