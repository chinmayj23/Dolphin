from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .features import apply_target, build_temporal_features
from .interestingness import LABEL_COL, numeric_transition, tail_cluster
from .io import ensure_dir, load_json, save_json
from .metrics import interestingness_metrics
from .plotting import save_distribution_plot, save_numeric_transition_plots
from .preprocessing import preprocess_data
from .rf_rules import run_rf_module
from .spans import build_transition_spans, build_value_spans
from .dolphin import run_dolphin, run_dolphin_binary


def run_pipeline(config_path: str | Path, workspace_root: str | Path | None = None) -> None:
    config_path = Path(config_path).resolve()
    root = Path(workspace_root).resolve() if workspace_root else config_path.parents[2]
    cfg = load_json(config_path)
    data_cfg = cfg["data"]
    id_col = data_cfg["id_col"]
    date_col = data_cfg["date_col"]
    data_path = _resolve(root, data_cfg["path"])
    output_root = ensure_dir(_resolve(root, cfg.get("output_dir", "outputs")))
    raw = pd.read_csv(data_path)
    raw, preprocessing_report = preprocess_data(raw, data_cfg, root)
    save_json(output_root / "preprocessing.json", preprocessing_report)
    summary = []

    for target_cfg in cfg["targets"]:
        target_name = target_cfg["name"]
        work, target_col, target_excludes = apply_target(raw, target_cfg)
        table, feature_names = build_temporal_features(
            work,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
            feature_cfg=cfg.get("feature_engineering", {}),
            exclude_cols=target_excludes,
        )
        target_dir = ensure_dir(output_root / target_name)

        if cfg.get("methods", {}).get("tail_cluster", {}).get("enabled", True):
            method_cfg = cfg["methods"]["tail_cluster"]
            labeled = tail_cluster(table, id_col, target_col, target_cfg, method_cfg)
            spans = build_value_spans(labeled, id_col, date_col, target_col, feature_names)
            _save_method_outputs(
                target_dir / "tail_cluster",
                labeled,
                spans,
                id_col,
                date_col,
                target_col,
                feature_names,
                cfg,
                summary,
                target_name,
                "tail_cluster",
            )

        if cfg.get("methods", {}).get("numeric_transition", {}).get("enabled", True):
            method_cfg = cfg["methods"]["numeric_transition"]
            labeled = numeric_transition(table, id_col, target_col, target_cfg, method_cfg)
            spans = build_transition_spans(labeled, id_col, date_col, target_col, feature_names)
            _save_method_outputs(
                target_dir / "numeric_transition",
                labeled,
                spans,
                id_col,
                date_col,
                target_col,
                feature_names,
                cfg,
                summary,
                target_name,
                "numeric_transition",
            )

        if cfg.get("methods", {}).get("dolphin", {}).get("enabled", True):
            method_cfg = cfg["methods"]["dolphin"]
            traj_metrics = run_dolphin(
                table=table,
                id_col=id_col,
                target_col=target_col,
                feature_names=feature_names,
                cfg=method_cfg,
                output_dir=target_dir / "dolphin",
            )
            summary.append(
                {
                    "target": target_name,
                    "method": "dolphin",
                    "rows": int(len(table)),
                    "interesting_rows": None,
                    "support": None,
                    "spans": None,
                    "interesting_spans": None,
                    "span_support": None,
                    "rf_balanced_accuracy": None,
                    "rf_f1": None,
                    "dolphin_entities": traj_metrics.get("n_entities"),
                    "dolphin_rules": traj_metrics.get("n_selected_rules"),
                    "dolphin_top_quality": traj_metrics.get("top_quality"),
                    "dolphin_top_divergence": traj_metrics.get("top_divergence"),
                }
            )

        if cfg.get("methods", {}).get("dolphin_binary", {}).get("enabled", False):
            method_cfg = cfg["methods"]["dolphin_binary"]
            binary_metrics = run_dolphin_binary(
                table=table,
                id_col=id_col,
                target_col=target_col,
                feature_names=feature_names,
                cfg=method_cfg,
                output_dir=target_dir / "dolphin_binary",
            )
            summary.append(
                {
                    "target": target_name,
                    "method": "dolphin_binary",
                    "rows": int(len(table)),
                    "interesting_rows": binary_metrics.get("n_interesting_entities"),
                    "support": binary_metrics.get("interesting_support"),
                    "spans": None,
                    "interesting_spans": None,
                    "span_support": None,
                    "rf_balanced_accuracy": binary_metrics.get("surrogate_balanced_accuracy"),
                    "rf_f1": binary_metrics.get("surrogate_f1"),
                    "dolphin_entities": binary_metrics.get("n_entities"),
                    "dolphin_rules": binary_metrics.get("n_selected_rules"),
                    "dolphin_top_quality": binary_metrics.get("top_quality"),
                    "dolphin_top_divergence": binary_metrics.get("top_divergence"),
                }
            )

    pd.DataFrame(summary).to_csv(output_root / "run_summary.csv", index=False)
    save_json(output_root / "run_summary.json", {"records": summary})


def _save_method_outputs(
    method_dir: Path,
    labeled: pd.DataFrame,
    spans: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    feature_names: list[str],
    cfg: dict,
    summary: list[dict],
    target_name: str,
    method_name: str,
) -> None:
    if method_dir.exists():
        shutil.rmtree(method_dir)
    method_dir = ensure_dir(method_dir)
    _slim_rows(labeled, id_col, date_col, target_col).to_csv(method_dir / "row_labels.csv", index=False)
    spans.to_csv(method_dir / "spans.csv", index=False)
    _transition_counts(labeled).to_csv(method_dir / "transition_counts.csv", index=False)
    _transition_summary(labeled).to_csv(method_dir / "transition_summary.csv", index=False)
    metrics = interestingness_metrics(labeled, spans, target_col)
    save_json(method_dir / "metrics.json", metrics)
    save_distribution_plot(labeled, target_col, method_dir / "interestingness_kde")
    if "transition_z" in labeled.columns:
        save_numeric_transition_plots(labeled, method_dir)
    rf_metrics = run_rf_module(
        spans,
        id_col,
        feature_names,
        cfg.get("split", {}),
        cfg.get("rf", {}),
        method_dir / "rf",
        method_name=method_name,
    )
    save_json(
        method_dir / "artifacts.json",
        {
            "row_labels": "row_labels.csv",
            "spans": "spans.csv",
            "transition_counts": "transition_counts.csv",
            "transition_summary": "transition_summary.csv",
            "metrics": "metrics.json",
            "kde": "interestingness_kde.png",
            "kde_svg": "interestingness_kde.svg",
            "kde_pdf": "interestingness_kde.pdf",
            "transition_delta_kde": "transition_delta_kde.png",
            "transition_z_kde": "transition_z_kde.png",
            "transition_score_kde": "transition_score_kde.png",
            "method_score_kde": "interesting_score_kde.png",
            "rf_metrics": "rf/rule_metrics.json",
            "rules": "rf/rules.csv",
            "forest_summary": "rf/forest_summary.csv",
            "forest_trees": "rf/forest_trees",
            "forest_tree_manifest": "rf/forest_trees/manifest.csv",
            "rule_kde_plots": "rf/rule_plots",
            "subgroup_distribution": "rf/subgroup_distribution.png",
            "subgroup_distribution_overlay": "rf/subgroup_distribution_overlay.png",
            "subgroup_distribution_axis_selection": "rf/subgroup_distribution_axis_selection.csv",
            "subgroup_distribution_rules": "rf/subgroup_distribution_rules.txt",
        },
    )
    summary.append(
        {
            "target": target_name,
            "method": method_name,
            "rows": metrics["n_rows"],
            "interesting_rows": metrics["n_interesting_rows"],
            "support": metrics["support"],
            "spans": metrics["n_spans"],
            "interesting_spans": metrics["n_interesting_spans"],
            "span_support": metrics["span_support"],
            "rf_balanced_accuracy": rf_metrics.get("balanced_accuracy"),
            "rf_f1": rf_metrics.get("f1"),
        }
    )


def _slim_rows(labeled: pd.DataFrame, id_col: str, date_col: str, target_col: str) -> pd.DataFrame:
    cols = [
        id_col,
        date_col,
        "_year",
        target_col,
        LABEL_COL,
        "interesting_score",
        "state",
        "next_year",
        "transition",
        "transition_probability",
        "next_target",
        "transition_delta",
        "expected_delta",
        "transition_residual",
        "transition_z",
        "transition_score",
        "method",
        "threshold_high",
        "threshold_low",
    ]
    return labeled[[c for c in cols if c in labeled.columns]].copy()


def _transition_counts(labeled: pd.DataFrame) -> pd.DataFrame:
    if "transition" not in labeled.columns:
        return pd.DataFrame()
    counts = (
        labeled.groupby(["transition", LABEL_COL], dropna=False)
        .size()
        .reset_index(name="n_rows")
        .sort_values([LABEL_COL, "n_rows"], ascending=[False, False])
    )
    counts["row_support"] = counts["n_rows"] / len(labeled) if len(labeled) else 0.0
    return counts


def _transition_summary(labeled: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["transition_delta", "transition_residual", "transition_z", "transition_score", "interesting_score"] if c in labeled.columns]
    if not cols:
        return pd.DataFrame()
    rows = []
    for label_value, group in labeled.groupby(LABEL_COL, dropna=False):
        rec = {"is_interesting": int(label_value), "n_rows": int(len(group))}
        for col in cols:
            values = pd.to_numeric(group[col], errors="coerce")
            rec[f"{col}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            rec[f"{col}_median"] = float(values.median()) if values.notna().any() else np.nan
            rec[f"{col}_min"] = float(values.min()) if values.notna().any() else np.nan
            rec[f"{col}_max"] = float(values.max()) if values.notna().any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).sort_values("is_interesting", ascending=False)


def _resolve(root: Path, path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path
