from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO / "src"))

from tsd.features import apply_target, build_temporal_features
from tsd.io import ensure_dir
from tsd.preprocessing import preprocess_data
from tsd.trajtrack import run_trajtrack_binary


SCENARIOS = {
    "q90_default": {},
    "q85_more_recall": {
        "anomaly_quantile": 0.85,
        "min_precision": 0.35,
        "min_lift": 1.4,
    },
    "q95_strict_tail": {
        "anomaly_quantile": 0.95,
        "min_positive_entities": 5,
        "min_precision": 0.5,
        "min_lift": 2.0,
    },
    "q90_high_precision": {
        "anomaly_quantile": 0.9,
        "min_precision": 0.6,
        "min_lift": 2.5,
    },
    "q90_compact_rules": {
        "anomaly_quantile": 0.9,
        "forest_min_trees": 4,
        "forest_max_trees": 8,
        "forest_min_questions_per_tree": 4,
        "forest_max_questions_per_tree": 10,
        "surrogate_min_samples_leaf": 6,
        "min_support": 0.03,
        "max_support": 0.25,
    },
    "q90_complex_rules": {
        "anomaly_quantile": 0.9,
        "forest_min_trees": 5,
        "forest_max_trees": 12,
        "forest_min_questions_per_tree": 8,
        "forest_max_questions_per_tree": 22,
        "surrogate_min_samples_leaf": 3,
        "min_support": 0.015,
        "max_support": 0.3,
    },
    "q90_relaxed_diversity": {
        "anomaly_quantile": 0.9,
        "max_rule_jaccard": 0.85,
        "min_rule_trajectory_distance": 0.15,
        "max_rule_trajectory_correlation": 0.998,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs" / "world_bank.json"))
    parser.add_argument("--search-seeds", type=int, default=150)
    parser.add_argument("--output-dir", default=str(REPO / "outputs" / "world_bank" / "trajtrack_binary_sweep"))
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw = pd.read_csv(WORKSPACE / cfg["data"]["path"])
    raw, preprocessing_report = preprocess_data(raw, cfg["data"], WORKSPACE)
    output_root = ensure_dir(Path(args.output_dir))
    (output_root / "preprocessing.json").write_text(json.dumps(preprocessing_report, indent=2), encoding="utf-8")

    id_col = cfg["data"]["id_col"]
    date_col = cfg["data"]["date_col"]
    results = []
    for target_cfg in cfg["targets"]:
        target_name = target_cfg["name"]
        if args.targets and target_name not in args.targets:
            continue
        work, target_col, target_excludes = apply_target(raw, target_cfg)
        table, feature_names = build_temporal_features(
            work,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
            feature_cfg=cfg["feature_engineering"],
            exclude_cols=target_excludes,
        )
        base = copy.deepcopy(cfg["methods"]["trajtrack_binary"])
        base["forest_search_seeds"] = args.search_seeds
        for scenario_name, overrides in SCENARIOS.items():
            if args.scenarios and scenario_name not in args.scenarios:
                continue
            scenario_cfg = copy.deepcopy(base)
            scenario_cfg.update(overrides)
            scenario_dir = output_root / target_name / scenario_name
            print(f"{target_name} / {scenario_name}", flush=True)
            metrics = run_trajtrack_binary(table, id_col, target_col, feature_names, scenario_cfg, scenario_dir)
            result = {
                "target": target_name,
                "scenario": scenario_name,
                **overrides,
                **metrics,
                **_rule_summary(scenario_dir),
            }
            results.append(result)
            pd.DataFrame(results).to_csv(output_root / "sweep_summary_partial.csv", index=False)

    summary = pd.DataFrame(results)
    summary.to_csv(output_root / "sweep_summary.csv", index=False)
    (output_root / "scenarios.json").write_text(json.dumps(SCENARIOS, indent=2), encoding="utf-8")
    _write_report(summary, output_root / "sweep_report.md")


def _rule_summary(output_dir: Path) -> dict:
    path = output_dir / "rules.csv"
    if not path.exists():
        return {}
    rules = pd.read_csv(path)
    if rules.empty:
        return {
            "mean_precision": None,
            "max_precision": None,
            "mean_lift": None,
            "max_lift": None,
            "mean_recall": None,
            "max_recall": None,
            "mean_conditions": 0.0,
            "max_conditions": 0,
            "mean_effect": None,
            "max_effect": None,
        }
    return {
        "mean_precision": float(rules["precision"].mean()),
        "max_precision": float(rules["precision"].max()),
        "mean_lift": float(rules["lift"].mean()),
        "max_lift": float(rules["lift"].max()),
        "mean_recall": float(rules["recall"].mean()),
        "max_recall": float(rules["recall"].max()),
        "mean_conditions": float(rules["n_conditions"].mean()),
        "max_conditions": int(rules["n_conditions"].max()),
        "mean_effect": float(rules["baseline_effect_size"].mean()),
        "max_effect": float(rules["baseline_effect_size"].max()),
        "top_rule": str(rules.iloc[0]["rule"]),
        "top_rule_entities": str(rules.iloc[0]["entities"]),
    }


def _write_report(summary: pd.DataFrame, path: Path) -> None:
    lines = ["# TrajTrack Binary Hyperparameter Sweep", ""]
    rank_cols = [
        "scenario",
        "n_interesting_entities",
        "n_selected_rules",
        "surrogate_balanced_accuracy",
        "surrogate_f1",
        "mean_precision",
        "max_precision",
        "mean_lift",
        "max_recall",
        "max_effect",
        "mean_conditions",
    ]
    for target, group in summary.groupby("target"):
        ranked = group.copy()
        ranked["selection_score"] = (
            ranked["surrogate_f1"].fillna(0)
            + 0.5 * ranked["mean_precision"].fillna(0)
            + 0.2 * ranked["n_selected_rules"].fillna(0).clip(upper=4)
            - 0.03 * ranked["mean_conditions"].fillna(0)
        )
        ranked = ranked.sort_values("selection_score", ascending=False)
        lines.extend([f"## {target}", ""])
        table = ranked[rank_cols].fillna("")
        lines.append("| " + " | ".join(rank_cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(rank_cols)) + " |")
        for row in table.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        best = ranked.iloc[0]
        lines.extend(
            [
                "",
                f"Selected sweep candidate: `{best['scenario']}`.",
                f"Top rule: {best.get('top_rule', '')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
