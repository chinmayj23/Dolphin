from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair copy-pasted CMIE CSV lines wrapped in quotes.")
    parser.add_argument("--input", default="transition_subgroup_discovery/datax.csv")
    parser.add_argument("--output", default="transition_subgroup_discovery/data/cmie_trajtrack_panel.csv")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    bad = 0
    expected = None
    with src.open("r", encoding="utf-8-sig", newline="") as fh_in, dst.open(
        "w", encoding="utf-8", newline=""
    ) as fh_out:
        writer = csv.writer(fh_out)
        for raw in fh_in:
            line = raw.rstrip("\r\n")
            if len(line) >= 2 and line[0] == '"' and line[-1] == '"':
                line = line[1:-1]
            parsed = next(csv.reader([line]))
            if expected is None:
                expected = len(parsed)
                writer.writerow(parsed)
                continue
            if len(parsed) != expected:
                bad += 1
                continue
            writer.writerow(parsed)
            rows += 1

    df = pd.read_csv(dst)
    print(f"wrote: {dst}")
    print(f"rows: {rows}; bad rows skipped: {bad}; shape: {df.shape}")
    print(f"entities: {df['HH_ID'].nunique()}")
    print(f"date range: {df['MONTH_SLOT_DATE'].min()} to {df['MONTH_SLOT_DATE'].max()}")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
