from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parent
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from tsd.features import apply_target, build_temporal_features
    from tsd.io import load_json
    from tsd.preprocessing import preprocess_data
    from tsd.trajtrack import (
        _build_trajectories,
        _entity_anomaly_scores,
        _save_rule_plots,
        _set_pretty_time_unit,
    )

    parser = argparse.ArgumentParser(description="Regenerate FLAIR trajectory plots from existing memberships.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace-root", default=str(workspace))
    parser.add_argument("--target", default=None)
    parser.add_argument("--method-dir", default="trajtrack")
    args = parser.parse_args()

    root = Path(args.workspace_root).resolve()
    cfg = load_json(Path(args.config).resolve())
    data_cfg = cfg["data"]
    data_path = Path(data_cfg["path"])
    if not data_path.is_absolute():
        data_path = root / data_path
    output_root = Path(cfg.get("output_dir", "outputs"))
    if not output_root.is_absolute():
        output_root = root / output_root

    raw = pd.read_csv(data_path)
    raw, _ = preprocess_data(raw, data_cfg, root)
    method_cfg = cfg["methods"]["trajtrack"]
    _set_pretty_time_unit(method_cfg.get("time_unit_label"))

    for target_cfg in cfg["targets"]:
        if args.target and target_cfg["name"] != args.target:
            continue
        output_dir = output_root / target_cfg["name"] / args.method_dir
        membership_path = output_dir / "membership.csv"
        if not membership_path.exists():
            print(f"Skipping {target_cfg['name']}: no membership at {membership_path}")
            continue

        work, target_col, target_excludes = apply_target(raw, target_cfg)
        table, feature_names = build_temporal_features(
            work,
            id_col=data_cfg["id_col"],
            date_col=data_cfg["date_col"],
            target_col=target_col,
            feature_cfg=cfg.get("feature_engineering", {}),
            exclude_cols=target_excludes,
        )
        trajectory_pack = _build_trajectories(
            table=table,
            id_col=data_cfg["id_col"],
            target_col=target_col,
            representation=str(method_cfg.get("representation", "linear")),
            grid_size=int(method_cfg.get("grid_size", 25)),
            min_points=int(method_cfg.get("min_points", 6)),
        )
        entity_ids = [str(x) for x in trajectory_pack["entity_ids"]]
        trajectories = trajectory_pack["trajectories"]
        centered = trajectories - trajectories.mean(axis=1, keepdims=True)
        baseline = centered.mean(axis=0)
        anomaly_scores = _entity_anomaly_scores(
            centered,
            baseline,
            str(method_cfg.get("anomaly_metric", "euclidean")),
        )

        membership = pd.read_csv(membership_path)
        membership["entity"] = membership["entity"].astype(str)
        membership = membership.set_index("entity").reindex(entity_ids).fillna(0).reset_index()
        rule_cols = [c for c in membership.columns if c.startswith("rule_")]
        rules = []
        for col in rule_cols:
            mask = membership[col].astype(int).to_numpy() == 1
            if mask.any():
                rules.append({"mask": mask})
        _save_rule_plots(
            rules,
            centered,
            baseline,
            anomaly_scores,
            trajectories.mean(axis=1),
            trajectory_pack["grid"],
            trajectory_pack["raw_lookup"],
            entity_ids,
            output_dir / "plots",
        )
        print(f"Regenerated {len(rules)} trajectory plot(s): {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
