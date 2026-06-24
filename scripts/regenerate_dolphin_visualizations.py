from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    sys.path.insert(0, str(repo / "src"))

    from tsd.features import apply_target, build_temporal_features
    from tsd.io import load_json
    from tsd.preprocessing import preprocess_data
    from tsd.dolphin import (
        _build_trajectories,
        _save_rule_plots,
        _save_syflow_distribution_overview,
    )

    cfg = load_json(repo / "configs" / "world_bank.json")
    data_cfg = cfg["data"]
    raw = pd.read_csv(workspace / data_cfg["path"])
    raw, _ = preprocess_data(raw, data_cfg, workspace)
    output_root = workspace / cfg["output_dir"]

    for target_cfg in cfg["targets"]:
        work, target_col, target_excludes = apply_target(raw, target_cfg)
        table, feature_names = build_temporal_features(
            work,
            id_col=data_cfg["id_col"],
            date_col=data_cfg["date_col"],
            target_col=target_col,
            feature_cfg=cfg.get("feature_engineering", {}),
            exclude_cols=target_excludes,
        )
        for method in ["dolphin", "dolphin_binary"]:
            if not cfg["methods"].get(method, {}).get("enabled", False):
                continue
            method_cfg = cfg["methods"][method]
            out_dir = output_root / target_cfg["name"] / method
            if not (out_dir / "rules.csv").exists():
                continue
            pack = _build_trajectories(
                table,
                data_cfg["id_col"],
                target_col,
                representation=str(method_cfg.get("representation", "linear")),
                grid_size=int(method_cfg.get("grid_size", 25)),
                min_points=int(method_cfg.get("min_points", 6)),
            )
            entity_ids = pack["entity_ids"]
            trajectories = pack["trajectories"]
            centered = trajectories - trajectories.mean(axis=1, keepdims=True)
            baseline = centered.mean(axis=0)
            membership = pd.read_csv(out_dir / "membership.csv").set_index("entity").reindex(entity_ids)
            rules_frame = pd.read_csv(out_dir / "rules.csv")
            anomaly = (
                pd.read_csv(out_dir / "anomaly_scores.csv")
                .set_index("entity")
                .reindex(entity_ids)["trajectory_anomaly_score"]
                .to_numpy(dtype=float)
            )
            rules = []
            for index, row in rules_frame.iterrows():
                rules.append(
                    {
                        "mask": membership[f"rule_{index + 1:02d}"].fillna(0).to_numpy(dtype=bool),
                        "quality": float(row["quality"]),
                        "support": float(row["support"]),
                        "selectors": [],
                        "rule_text": str(row["rule"]),
                    }
                )
            _save_rule_plots(
                rules,
                centered,
                baseline,
                anomaly,
                trajectories.mean(axis=1),
                pack["grid"],
                pack["raw_lookup"],
                out_dir / "plots",
            )
            _save_syflow_distribution_overview(
                rules,
                entity_ids,
                anomaly,
                trajectories,
                out_dir / "subgroup_distribution",
                out_dir / "subgroup_distribution_rules.txt",
            )


if __name__ == "__main__":
    main()
