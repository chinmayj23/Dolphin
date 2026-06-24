from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    sys.path.insert(0, str(repo / "src"))
    from tsd.trajtrack import _save_subgroup_evidence_summary

    raw = pd.read_csv(workspace / "wbd" / "world_bank_development_indicators.csv")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["gdp_per_capita"] = pd.to_numeric(raw["GDP_current_US"], errors="coerce") / pd.to_numeric(
        raw["population"], errors="coerce"
    )
    output_root = repo / "outputs" / "world_bank"
    targets = {
        "life_expectancy": "life_expectancy_at_birth",
        "gdp_per_capita": "gdp_per_capita",
    }
    for target_name, target_col in targets.items():
        out_dir = output_root / target_name / "trajtrack"
        membership = pd.read_csv(out_dir / "membership.csv")
        rules_frame = pd.read_csv(out_dir / "rules.csv")
        entity_ids = membership["entity"].astype(str).tolist()
        trajectories = []
        for entity in entity_ids:
            values = (
                raw.loc[raw["country"].astype(str) == entity, ["date", target_col]]
                .dropna()
                .sort_values("date")[target_col]
                .to_numpy(dtype=float)
            )
            trajectories.append(np.linspace(values[0], values[-1], 25))
        rules = []
        for index, row in rules_frame.iterrows():
            rules.append(
                {
                    "mask": membership[f"rule_{index + 1:02d}"].to_numpy(dtype=bool),
                    "rule_text": str(row["rule"]),
                    "baseline_effect_size": float(row["baseline_effect_size"]),
                    "quality": float(row["quality"]),
                }
            )
        _save_subgroup_evidence_summary(
            rules,
            entity_ids,
            np.asarray(trajectories),
            out_dir / "subgroup_evidence_summary",
            out_dir / "subgroup_evidence_summary.txt",
        )


if __name__ == "__main__":
    main()
