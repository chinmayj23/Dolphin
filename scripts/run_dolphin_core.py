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
    from tsd.io import ensure_dir, load_json
    from tsd.preprocessing import preprocess_data
    from tsd.dolphin import run_dolphin, run_dolphin_binary

    parser = argparse.ArgumentParser(description="Run DOLPHIN methods from a JSON config.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--workspace-root", default=str(workspace))
    parser.add_argument("--method", choices=["dolphin", "dolphin_binary", "both"], default="dolphin")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    root = Path(args.workspace_root).resolve()
    cfg = load_json(config_path)
    data_cfg = cfg["data"]
    data_path = Path(data_cfg["path"])
    if not data_path.is_absolute():
        data_path = root / data_path
    output_root = Path(cfg.get("output_dir", "outputs"))
    if not output_root.is_absolute():
        output_root = root / output_root
    output_root = ensure_dir(output_root)

    raw = pd.read_csv(data_path)
    raw, report = preprocess_data(raw, data_cfg, root)
    if report:
        ensure_dir(output_root).joinpath("preprocessing.json").write_text(
            pd.Series(report).to_json(indent=2),
            encoding="utf-8",
        )

    summaries = []
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
        if args.method in {"dolphin", "both"} and cfg["methods"].get("dolphin", {}).get("enabled", True):
            print(f"Running DOLPHIN: {target_cfg['name']}", flush=True)
            metrics = run_dolphin(
                table=table,
                id_col=data_cfg["id_col"],
                target_col=target_col,
                feature_names=feature_names,
                cfg=cfg["methods"]["dolphin"],
                output_dir=output_root / target_cfg["name"] / "dolphin",
            )
            summaries.append({"target": target_cfg["name"], "method": "dolphin", **metrics})
        if args.method in {"dolphin_binary", "both"} and cfg["methods"].get("dolphin_binary", {}).get("enabled", False):
            print(f"Running DOLPHIN Binary: {target_cfg['name']}", flush=True)
            metrics = run_dolphin_binary(
                table=table,
                id_col=data_cfg["id_col"],
                target_col=target_col,
                feature_names=feature_names,
                cfg=cfg["methods"]["dolphin_binary"],
                output_dir=output_root / target_cfg["name"] / "dolphin_binary",
            )
            summaries.append({"target": target_cfg["name"], "method": "dolphin_binary", **metrics})

    if summaries:
        pd.DataFrame(summaries).to_csv(output_root / "run_summary.csv", index=False)


if __name__ == "__main__":
    main()
