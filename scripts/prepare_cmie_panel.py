from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


MAX_EDU_ORDER = {
    "Unknown": float("nan"),
    "None or Primary": 1.0,
    "Class 10": 2.0,
    "Class 12 / Diploma": 3.0,
    "Graduate and Above": 4.0,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare repaired CMIE panel for DOLPHIN.")
    parser.add_argument("--input", default="transition_subgroup_discovery/data/cmie_dolphin_panel.csv")
    parser.add_argument("--output", default="transition_subgroup_discovery/data/cmie_dolphin_panel.csv")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    df = pd.read_csv(src)
    if "MAX_EDU" in df.columns:
        df["MAX_EDU_LEVEL"] = df["MAX_EDU"].astype("string").map(MAX_EDU_ORDER).astype(float)
    categorical = [col for col in ["dominant_source", "income_quintile"] if col in df.columns]
    if categorical:
        dummies = pd.get_dummies(
            df[categorical].astype("string").fillna("Unknown"),
            prefix=categorical,
            dtype=float,
        )
        df = pd.concat([df.drop(columns=categorical), dummies], axis=1)
    drop_cols = [col for col in ["STATE", "REGION_TYPE", "MAX_EDU"] if col in df.columns]
    df = df.drop(columns=drop_cols)

    for col in df.columns:
        if col != "MONTH_SLOT_DATE":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(f"wrote: {dst}")
    n_dummy = sum(col.startswith(("dominant_source_", "income_quintile_")) for col in df.columns)
    print(f"shape: {df.shape}; encoded MAX_EDU_LEVEL: {'MAX_EDU_LEVEL' in df.columns}; dummy columns: {n_dummy}")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
