from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
sys.path.insert(0, str(REPO / "src"))

from tsd.features import apply_target, build_temporal_features
from tsd.io import ensure_dir
from tsd.preprocessing import preprocess_data
from tsd.dolphin import (
    _build_entity_feature_table,
    _build_trajectories,
    _entity_anomaly_scores,
    run_dolphin,
)


SCENARIOS = {
    "baseline_compact": {},
    "quality_forest": {
        "forest_selection_mode": "quality",
    },
    "compact_low_support": {
        "min_support": 0.02,
        "max_support": 0.25,
    },
    "quality_low_support": {
        "forest_selection_mode": "quality",
        "min_support": 0.02,
        "max_support": 0.25,
    },
    "quality_complex": {
        "forest_selection_mode": "quality",
        "forest_min_trees": 3,
        "forest_max_trees": 10,
        "forest_min_questions_per_tree": 4,
        "forest_max_questions_per_tree": 16,
        "surrogate_min_samples_leaf": 5,
        "min_support": 0.03,
        "max_support": 0.30,
    },
    "quality_complex_tail": {
        "forest_selection_mode": "quality",
        "forest_min_trees": 3,
        "forest_max_trees": 10,
        "forest_min_questions_per_tree": 6,
        "forest_max_questions_per_tree": 18,
        "surrogate_min_samples_leaf": 4,
        "min_support": 0.02,
        "max_support": 0.25,
    },
    "quality_stable_forest": {
        "forest_selection_mode": "quality",
        "forest_min_trees": 5,
        "forest_max_trees": 10,
        "forest_min_questions_per_tree": 4,
        "forest_max_questions_per_tree": 14,
        "surrogate_min_samples_leaf": 8,
        "min_support": 0.03,
        "max_support": 0.30,
    },
    "quality_deep_stable": {
        "forest_selection_mode": "quality",
        "forest_min_trees": 5,
        "forest_max_trees": 10,
        "forest_min_questions_per_tree": 6,
        "forest_max_questions_per_tree": 18,
        "surrogate_min_samples_leaf": 4,
        "min_support": 0.02,
        "max_support": 0.25,
    },
    "quality_complex_relaxed_effect": {
        "forest_selection_mode": "quality",
        "forest_min_trees": 3,
        "forest_max_trees": 10,
        "forest_min_questions_per_tree": 4,
        "forest_max_questions_per_tree": 16,
        "surrogate_min_samples_leaf": 5,
        "min_support": 0.03,
        "max_support": 0.30,
        "min_baseline_effect_size": 0.35,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO / "configs" / "world_bank.json"))
    parser.add_argument("--search-seeds", type=int, default=150)
    parser.add_argument("--output-dir", default=str(REPO / "outputs" / "world_bank" / "dolphin_sweep"))
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    raw = pd.read_csv(WORKSPACE / cfg["data"]["path"])
    raw, preprocessing_report = preprocess_data(raw, cfg["data"], WORKSPACE)
    output_root = ensure_dir(Path(args.output_dir))
    (output_root / "preprocessing.json").write_text(
        json.dumps(preprocessing_report, indent=2), encoding="utf-8"
    )
    id_col = cfg["data"]["id_col"]
    date_col = cfg["data"]["date_col"]
    results = []
    profiles = []

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
        base = copy.deepcopy(cfg["methods"]["dolphin"])
        base["forest_search_seeds"] = args.search_seeds
        profiles.extend(_profile_target(table, id_col, target_col, feature_names, base, target_name))

        for scenario_name, overrides in SCENARIOS.items():
            if args.scenarios and scenario_name not in args.scenarios:
                continue
            scenario_cfg = copy.deepcopy(base)
            scenario_cfg.update(overrides)
            scenario_dir = output_root / target_name / scenario_name
            metrics = run_dolphin(table, id_col, target_col, feature_names, scenario_cfg, scenario_dir)
            result = {
                "target": target_name,
                "scenario": scenario_name,
                **overrides,
                **metrics,
            }
            result.update(_rule_tail_summary(scenario_dir))
            results.append(result)
            pd.DataFrame(results).to_csv(output_root / "sweep_summary_partial.csv", index=False)

    summary = pd.DataFrame(results)
    summary.to_csv(output_root / "sweep_summary.csv", index=False)
    pd.DataFrame(profiles).to_csv(output_root / "data_profile.csv", index=False)
    (output_root / "scenarios.json").write_text(json.dumps(SCENARIOS, indent=2), encoding="utf-8")
    _write_report(summary, pd.DataFrame(profiles), output_root / "sweep_report.md")


def _profile_target(
    table: pd.DataFrame,
    id_col: str,
    target_col: str,
    feature_names: list[str],
    cfg: dict,
    target_name: str,
) -> list[dict]:
    pack = _build_trajectories(
        table,
        id_col,
        target_col,
        cfg.get("representation", "linear"),
        int(cfg.get("grid_size", 25)),
        int(cfg.get("min_points", 6)),
    )
    centered = pack["trajectories"] - pack["trajectories"].mean(axis=1, keepdims=True)
    anomaly = _entity_anomaly_scores(centered, centered.mean(axis=0), cfg.get("anomaly_metric", "euclidean"))
    features = _build_entity_feature_table(table, id_col, feature_names).set_index(id_col).reindex(pack["entity_ids"])
    rows = []
    for feature in features.columns:
        values = pd.to_numeric(features[feature], errors="coerce")
        rows.append(
            {
                "target": target_name,
                "record_type": "feature",
                "name": feature,
                "n_entities": len(features),
                "missing_fraction": float(values.isna().mean()),
                "n_unique": int(values.nunique(dropna=True)),
                "value": None,
            }
        )
    for quantile in (0.5, 0.75, 0.9, 0.95, 0.975, 0.99):
        rows.append(
            {
                "target": target_name,
                "record_type": "anomaly_quantile",
                "name": f"q{quantile:g}",
                "n_entities": len(anomaly),
                "missing_fraction": None,
                "n_unique": None,
                "value": float(np.quantile(anomaly, quantile)),
            }
        )
    return rows


def _rule_tail_summary(output_dir: Path) -> dict:
    rules_path = output_dir / "rules.csv"
    membership_path = output_dir / "membership.csv"
    anomaly_path = output_dir / "anomaly_scores.csv"
    if not rules_path.exists() or not membership_path.exists() or not anomaly_path.exists():
        return {}
    rules = pd.read_csv(rules_path)
    if rules.empty:
        return {
            "max_rule_conditions": 0,
            "mean_rule_conditions": 0.0,
            "best_rule_mean_anomaly_percentile": None,
            "best_rule_top_10pct_share": None,
        }
    membership = pd.read_csv(membership_path).set_index("entity")
    anomaly = pd.read_csv(anomaly_path).set_index("entity")["trajectory_anomaly_score"]
    percentiles = anomaly.rank(method="average", pct=True)
    top_cutoff = float(anomaly.quantile(0.9))
    means = []
    top_shares = []
    for col in membership.columns:
        members = membership.index[membership[col].astype(bool)]
        member_scores = anomaly.reindex(members).dropna()
        means.append(float(percentiles.reindex(members).mean()))
        top_shares.append(float((member_scores >= top_cutoff).mean()))
    return {
        "max_rule_conditions": int(rules["n_conditions"].max()),
        "mean_rule_conditions": float(rules["n_conditions"].mean()),
        "best_rule_mean_anomaly_percentile": max(means),
        "best_rule_top_10pct_share": max(top_shares),
        "max_baseline_effect_size": float(rules["baseline_effect_size"].max()),
    }


def _write_report(summary: pd.DataFrame, profile: pd.DataFrame, path: Path) -> None:
    lines = ["# DOLPHIN Hyperparameter Sweep", ""]
    groups = summary.groupby("target") if "target" in summary.columns else []
    for target, group in groups:
        ranked = group.sort_values(
            ["best_rule_top_10pct_share", "n_selected_rules", "max_rule_conditions"],
            ascending=[False, False, False],
        )
        columns = [
            "scenario",
            "n_candidate_rules",
            "n_selected_rules",
            "max_rule_conditions",
            "best_rule_mean_anomaly_percentile",
            "best_rule_top_10pct_share",
            "max_baseline_effect_size",
            "surrogate_test_r2",
            "n_selected_trees",
        ]
        table = ranked[columns].fillna("")
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in table.itertuples(index=False, name=None)]
        lines.extend([f"## {target}", "", header, separator, *rows, ""])
    usable = profile[(profile["record_type"] == "feature") & (profile["missing_fraction"] <= 0.5)]
    lines.extend([
        "## Data profile",
        "",
        f"Features retained at the configured 50% missingness threshold: {len(usable)} target-feature rows.",
        "",
        "The sweep ranks tail concentration separately from rule count. More rules are not treated as better unless they remain trajectory-distinct.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
