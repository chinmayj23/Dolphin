from __future__ import annotations

import copy
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
    from tsd.trajtrack import run_trajtrack

    cfg = load_json(repo / "configs" / "cmie.json")
    cfg["data"]["path"] = str(repo / "data" / "cmie_trajtrack_panel.csv")
    output_root = ensure_dir(repo / "outputs" / "cmie_sweep")

    scenarios = [
        {
            "name": "compact_1_3_6",
            "lags": [1, 3, 6],
            "windows": [3, 6],
            "max_depth": 3,
            "min_support": 0.05,
            "max_questions": 10,
            "min_distance": 0.35,
        },
        {
            "name": "recommended_1_3_6_12",
            "lags": [1, 3, 6, 12],
            "windows": [3, 6, 12],
            "max_depth": 3,
            "min_support": 0.04,
            "max_questions": 12,
            "min_distance": 0.35,
        },
        {
            "name": "diverse_1_3_6_12",
            "lags": [1, 3, 6, 12],
            "windows": [3, 6, 12],
            "max_depth": 3,
            "min_support": 0.04,
            "max_questions": 12,
            "min_distance": 0.50,
        },
    ]

    data_path = Path(cfg["data"]["path"])
    if not data_path.is_absolute():
        data_path = workspace / data_path
    raw = pd.read_csv(data_path)
    target_cfg = cfg["targets"][0]
    rows = []
    for scenario in scenarios:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["feature_engineering"]["lags"] = scenario["lags"]
        run_cfg["feature_engineering"]["windows"] = scenario["windows"]
        run_cfg["methods"]["trajtrack"]["surrogate_max_depth"] = scenario["max_depth"]
        run_cfg["methods"]["trajtrack"]["min_support"] = scenario["min_support"]
        run_cfg["methods"]["trajtrack"]["forest_max_questions_per_tree"] = scenario["max_questions"]
        run_cfg["methods"]["trajtrack"]["min_rule_trajectory_distance"] = scenario["min_distance"]
        run_cfg["methods"]["trajtrack"]["forest_search_seeds"] = 80
        run_cfg["methods"]["trajtrack"]["time_unit_label"] = "month"

        work, target_col, target_excludes = apply_target(raw, target_cfg)
        table, feature_names = build_temporal_features(
            work,
            id_col=run_cfg["data"]["id_col"],
            date_col=run_cfg["data"]["date_col"],
            target_col=target_col,
            feature_cfg=run_cfg["feature_engineering"],
            exclude_cols=target_excludes,
        )
        out_dir = output_root / scenario["name"] / target_cfg["name"] / "trajtrack"
        print(f"Running {scenario['name']} with {len(feature_names)} temporal features", flush=True)
        metrics = run_trajtrack(
            table=table,
            id_col=run_cfg["data"]["id_col"],
            target_col=target_col,
            feature_names=feature_names,
            cfg=run_cfg["methods"]["trajtrack"],
            output_dir=out_dir,
        )
        rules_path = out_dir / "rules.csv"
        rules = pd.read_csv(rules_path) if rules_path.exists() else pd.DataFrame()
        row = {
            "scenario": scenario["name"],
            "n_temporal_features": len(feature_names),
            "lags": ",".join(map(str, scenario["lags"])),
            "windows": ",".join(map(str, scenario["windows"])),
            "max_depth": scenario["max_depth"],
            "min_support": scenario["min_support"],
            "max_questions": scenario["max_questions"],
            "min_distance": scenario["min_distance"],
            **metrics,
            "mean_rule_complexity": float(rules["complexity"].mean()) if "complexity" in rules else None,
            "mean_precision": float(rules["precision"].mean()) if "precision" in rules else None,
            "mean_lift": float(rules["lift"].mean()) if "lift" in rules else None,
            "mean_support": float(rules["support"].mean()) if "support" in rules else None,
            "mean_divergence": float(rules["divergence"].mean()) if "divergence" in rules else None,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(output_root / "summary.csv", index=False)

    print(pd.DataFrame(rows).sort_values(["selected_separation_quality", "mean_rule_complexity"], ascending=[False, True]))
    print(f"Wrote {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
