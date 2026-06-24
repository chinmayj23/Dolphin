from __future__ import annotations

import shutil
import textwrap
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree, plot_tree

from .io import ensure_dir, save_json
from .tree_viz import save_graphviz_style_tree

try:
    from scipy.interpolate import make_interp_spline
    from scipy.stats import wasserstein_distance
except Exception:  # pragma: no cover - fallback for minimal environments
    make_interp_spline = None
    wasserstein_distance = None


_PRETTY_TIME_UNIT = "year"


def _set_pretty_time_unit(value: str | None) -> None:
    global _PRETTY_TIME_UNIT
    if value:
        _PRETTY_TIME_UNIT = str(value)


@dataclass(frozen=True)
class Selector:
    feature: str
    op: str
    threshold: float

    def apply(self, X: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(X[self.feature], errors="coerce")
        if self.op == "<=":
            return (values <= self.threshold).fillna(False).to_numpy()
        if self.op == ">":
            return (values > self.threshold).fillna(False).to_numpy()
        raise ValueError(f"Unsupported operator: {self.op}")

    def text(self) -> str:
        return f"{_pretty_feature_name(self.feature)} {self.op} {self.threshold:.4g}"


@dataclass
class TreeEnsemble:
    estimators_: list[DecisionTreeRegressor]


def run_dolphin(
    table: pd.DataFrame,
    id_col: str,
    target_col: str,
    feature_names: list[str],
    cfg: dict,
    output_dir: Path,
) -> dict:
    _set_pretty_time_unit(cfg.get("time_unit_label"))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir = ensure_dir(output_dir)
    trajectory_pack = _build_trajectories(
        table=table,
        id_col=id_col,
        target_col=target_col,
        representation=str(cfg.get("representation", "linear")),
        grid_size=int(cfg.get("grid_size", 25)),
        min_points=int(cfg.get("min_points", 6)),
    )
    entity_ids = trajectory_pack["entity_ids"]
    trajectories = trajectory_pack["trajectories"]
    grid = trajectory_pack["grid"]
    raw_lookup = trajectory_pack["raw_lookup"]
    centered = trajectories - trajectories.mean(axis=1, keepdims=True)
    baseline = centered.mean(axis=0)
    anomaly_scores = _entity_anomaly_scores(centered, baseline, str(cfg.get("anomaly_metric", "euclidean")))

    entity_features = _build_entity_feature_table(table, id_col, feature_names)
    entity_features = entity_features.set_index(id_col).reindex(entity_ids).reset_index()
    X = entity_features[[c for c in entity_features.columns if c != id_col]].apply(pd.to_numeric, errors="coerce")
    keep = X.isna().mean(axis=0) < float(cfg.get("max_missing_frac", 0.7))
    X = X.loc[:, keep]
    X = X.fillna(X.median(numeric_only=True))

    forest, selected_tree_indices, forest_metrics = _fit_compact_surrogate_forest(
        X,
        y=anomaly_scores,
        centered=centered,
        baseline=baseline,
        rule_selection_cfg=cfg,
        seed=int(cfg.get("seed", 42)),
        test_size=float(cfg.get("surrogate_test_size", 0.25)),
        n_search_seeds=int(cfg.get("forest_search_seeds", 1000)),
        min_trees=int(cfg.get("forest_min_trees", 1)),
        max_trees=int(cfg.get("forest_max_trees", 10)),
        min_questions=int(cfg.get("forest_min_questions_per_tree", 2)),
        max_questions=int(cfg.get("forest_max_questions_per_tree", 10)),
        max_depth=int(cfg.get("surrogate_max_depth", 4)),
        min_samples_leaf=int(cfg.get("surrogate_min_samples_leaf", 8)),
        performance_quantile=float(cfg.get("forest_performance_quantile", 0.99)),
        selection_mode=str(cfg.get("forest_selection_mode", "compact")),
        quality_tolerance=float(cfg.get("forest_quality_tolerance", 0.05)),
        rule_min_support=float(cfg.get("min_support", 0.05)),
        rule_max_support=float(cfg.get("max_support", 0.3)),
        support_exponent=float(cfg.get("support_exponent", 0.25)),
        concentration_exponent=float(cfg.get("concentration_exponent", 0.5)),
        max_concentration_ratio=float(cfg.get("max_concentration_ratio", 4.0)),
        output_dir=output_dir,
    )
    _save_surrogate_forest_trees(
        forest,
        selected_tree_indices,
        [_pretty_feature_name(c) for c in X.columns],
        X.to_numpy(dtype=float),
        entity_ids,
        anomaly_scores,
        centered,
        trajectories,
        target_col,
        high_anomaly_percentile=float(cfg.get("leaf_high_anomaly_percentile", 0.8)),
        min_baseline_effect_size=float(cfg.get("min_baseline_effect_size", 0.5)),
        output_dir=output_dir / "forest_trees",
    )
    candidates = _surrogate_rules_from_forest(
        forest,
        selected_tree_indices,
        X,
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
    )
    scored = _score_candidate_rules(
        candidates,
        anomaly_scores,
        centered,
        baseline,
        divergence_metric=str(cfg.get("divergence_metric", "euclidean")),
        support_exponent=float(cfg.get("support_exponent", 0.25)),
        concentration_exponent=float(cfg.get("concentration_exponent", 0.5)),
        max_concentration_ratio=float(cfg.get("max_concentration_ratio", 4.0)),
        min_baseline_effect_size=float(cfg.get("min_baseline_effect_size", 0.25)),
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
        evaluation_anomaly_quantile=float(cfg.get("evaluation_anomaly_quantile", 0.9)),
    )
    pareto = _pareto_candidates(scored, cfg)
    selection_pool = [dict(rule, pareto_optimal=True) for rule in pareto]
    selected = _weighted_covering(
        selection_pool,
        anomaly_scores,
        centered,
        baseline,
        top_k=int(cfg.get("top_k", 10)),
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
        coverage_decay=float(cfg.get("coverage_decay", 0.1)),
        support_exponent=float(cfg.get("support_exponent", 0.25)),
        concentration_exponent=float(cfg.get("concentration_exponent", 0.5)),
        max_concentration_ratio=float(cfg.get("max_concentration_ratio", 4.0)),
        divergence_metric=str(cfg.get("divergence_metric", "euclidean")),
        max_rule_jaccard=float(cfg.get("max_rule_jaccard", 0.7)),
        min_rule_trajectory_distance=float(cfg.get("min_rule_trajectory_distance", 0.25)),
        max_rule_trajectory_correlation=float(cfg.get("max_rule_trajectory_correlation", 0.995)),
    )

    rules_df = _rules_to_frame(selected, entity_ids)
    rules_df.to_csv(output_dir / "rules.csv", index=False)
    _save_anomaly_scores(entity_ids, anomaly_scores, output_dir / "anomaly_scores.csv")
    (output_dir / "rules_natural_language.txt").write_text(
        _natural_language_rules(rules_df),
        encoding="utf-8",
    )
    _save_membership(selected, entity_ids, output_dir / "membership.csv")
    _save_rule_plots(
        selected,
        centered,
        baseline,
        anomaly_scores,
        trajectories.mean(axis=1),
        grid,
        raw_lookup,
        entity_ids,
        output_dir / "plots",
    )
    _save_syflow_distribution_overview(
        selected,
        entity_ids,
        anomaly_scores,
        trajectories,
        output_dir / "subgroup_distribution",
        output_dir / "subgroup_distribution_rules.txt",
    )
    _save_trajectory_pca_distribution(
        selected,
        entity_ids,
        centered,
        output_dir / "trajectory_pca_distribution",
        output_dir / "trajectory_pca_distribution_rules.txt",
    )
    _save_trajectory_signature_distribution(
        selected,
        entity_ids,
        trajectories,
        output_dir / "trajectory_signature_distribution",
        output_dir / "trajectory_signature_distribution_rules.txt",
    )
    _save_fixed_trajectory_metric_distributions(
        selected,
        entity_ids,
        centered,
        baseline,
        trajectories,
        output_dir,
    )
    _save_subgroup_evidence_summary(
        selected,
        entity_ids,
        trajectories,
        output_dir / "subgroup_evidence_summary",
        output_dir / "subgroup_evidence_summary.txt",
    )

    metrics = {
        "n_entities": int(len(entity_ids)),
        "n_candidate_rules": int(len(candidates)),
        "n_pareto_rules": int(len(pareto)),
        "n_selection_pool_rules": int(len(selection_pool)),
        "n_selected_rules": int(len(selected)),
        "grid_size": int(len(grid)),
        "trajectory_centering": "entity_mean",
        "anomaly_score_mean": float(np.mean(anomaly_scores)),
        "anomaly_score_max": float(np.max(anomaly_scores)),
        "support_exponent": float(cfg.get("support_exponent", 0.25)),
        "concentration_exponent": float(cfg.get("concentration_exponent", 0.5)),
        "min_baseline_effect_size": float(cfg.get("min_baseline_effect_size", 0.25)),
        "max_rule_jaccard": float(cfg.get("max_rule_jaccard", 0.7)),
        "min_rule_trajectory_distance": float(cfg.get("min_rule_trajectory_distance", 0.25)),
        "max_rule_trajectory_correlation": float(cfg.get("max_rule_trajectory_correlation", 0.995)),
        "evaluation_anomaly_quantile": float(cfg.get("evaluation_anomaly_quantile", 0.9)),
        **forest_metrics,
        "top_quality": float(rules_df["quality"].max()) if not rules_df.empty else None,
        "top_divergence": float(rules_df["divergence"].max()) if not rules_df.empty else None,
    }
    save_json(output_dir / "metrics.json", metrics)
    save_json(
        output_dir / "artifacts.json",
        {
            "rules": "rules.csv",
            "natural_language_rules": "rules_natural_language.txt",
            "anomaly_scores": "anomaly_scores.csv",
            "forest_summary": "forest_summary.csv",
            "forest_selection": "forest_selection.csv",
            "forest_trees": "forest_trees",
            "forest_tree_manifest": "forest_trees/manifest.csv",
            "forest_performance_questions": "performance_vs_questions.png",
            "subgroup_distribution": "subgroup_distribution.png",
            "subgroup_distribution_overlay": "subgroup_distribution_overlay.png",
            "subgroup_distribution_axis_selection": "subgroup_distribution_axis_selection.csv",
            "subgroup_distribution_rules": "subgroup_distribution_rules.txt",
            "trajectory_pca_distribution": "trajectory_pca_distribution.png",
            "trajectory_pca_distribution_rules": "trajectory_pca_distribution_rules.txt",
            "trajectory_pca_distribution_components": "trajectory_pca_distribution_components.csv",
            "trajectory_signature_distribution": "trajectory_signature_distribution.png",
            "trajectory_signature_distribution_rules": "trajectory_signature_distribution_rules.txt",
            "trajectory_signature_distribution_axis_selection": "trajectory_signature_distribution_axis_selection.csv",
            "trajectory_baseline_deviation_distribution": "trajectory_baseline_deviation_distribution.png",
            "trajectory_baseline_deviation_distribution_rules": "trajectory_baseline_deviation_distribution_rules.txt",
            "trajectory_mean_abs_change_distribution": "trajectory_mean_abs_change_distribution.png",
            "trajectory_mean_abs_change_distribution_rules": "trajectory_mean_abs_change_distribution_rules.txt",
            "subgroup_evidence_summary": "subgroup_evidence_summary.png",
            "subgroup_evidence_summary_key": "subgroup_evidence_summary.txt",
            "membership": "membership.csv",
            "metrics": "metrics.json",
            "plots": "plots",
        },
    )
    return metrics


def run_dolphin_binary(
    table: pd.DataFrame,
    id_col: str,
    target_col: str,
    feature_names: list[str],
    cfg: dict,
    output_dir: Path,
) -> dict:
    _set_pretty_time_unit(cfg.get("time_unit_label"))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir = ensure_dir(output_dir)
    trajectory_pack = _build_trajectories(
        table=table,
        id_col=id_col,
        target_col=target_col,
        representation=str(cfg.get("representation", "linear")),
        grid_size=int(cfg.get("grid_size", 25)),
        min_points=int(cfg.get("min_points", 6)),
    )
    entity_ids = trajectory_pack["entity_ids"]
    trajectories = trajectory_pack["trajectories"]
    grid = trajectory_pack["grid"]
    raw_lookup = trajectory_pack["raw_lookup"]
    centered = trajectories - trajectories.mean(axis=1, keepdims=True)
    baseline = centered.mean(axis=0)
    anomaly_scores = _entity_anomaly_scores(centered, baseline, str(cfg.get("anomaly_metric", "euclidean")))
    labels, label_info = _make_binary_trajectory_labels(centered, baseline, anomaly_scores, cfg)

    entity_features = _build_entity_feature_table(table, id_col, feature_names)
    entity_features = entity_features.set_index(id_col).reindex(entity_ids).reset_index()
    X = entity_features[[c for c in entity_features.columns if c != id_col]].apply(pd.to_numeric, errors="coerce")
    keep = X.isna().mean(axis=0) < float(cfg.get("max_missing_frac", 0.7))
    X = X.loc[:, keep]
    X = X.fillna(X.median(numeric_only=True))

    forest, selected_tree_indices, forest_metrics = _fit_binary_surrogate_forest(
        X,
        labels=labels,
        anomaly_scores=anomaly_scores,
        centered=centered,
        baseline=baseline,
        rule_selection_cfg=cfg,
        seed=int(cfg.get("seed", 42)),
        test_size=float(cfg.get("surrogate_test_size", 0.25)),
        n_search_seeds=int(cfg.get("forest_search_seeds", 1000)),
        min_trees=int(cfg.get("forest_min_trees", 1)),
        max_trees=int(cfg.get("forest_max_trees", 10)),
        min_questions=int(cfg.get("forest_min_questions_per_tree", 2)),
        max_questions=int(cfg.get("forest_max_questions_per_tree", 10)),
        min_samples_leaf=int(cfg.get("surrogate_min_samples_leaf", 8)),
        quality_tolerance=float(cfg.get("forest_quality_tolerance", 0.05)),
        accuracy_tolerance=float(cfg.get("forest_accuracy_tolerance", 0.05)),
        selection_mode=str(cfg.get("forest_selection_mode", "binary_diversity_tolerance")),
        rule_min_support=float(cfg.get("min_support", 0.05)),
        rule_max_support=float(cfg.get("max_support", 0.3)),
        output_dir=output_dir,
    )
    _save_surrogate_forest_trees(
        forest,
        selected_tree_indices,
        [_pretty_feature_name(c) for c in X.columns],
        X.to_numpy(dtype=float),
        entity_ids,
        anomaly_scores,
        centered,
        trajectories,
        target_col,
        high_anomaly_percentile=float(cfg.get("leaf_high_anomaly_percentile", 0.8)),
        min_baseline_effect_size=float(cfg.get("min_baseline_effect_size", 0.5)),
        output_dir=output_dir / "forest_trees",
    )
    candidates = _surrogate_rules_from_forest(
        forest,
        selected_tree_indices,
        X,
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
    )
    scored = _score_binary_candidate_rules(
        candidates,
        labels,
        anomaly_scores,
        centered,
        baseline,
        divergence_metric=str(cfg.get("divergence_metric", "euclidean")),
        support_exponent=float(cfg.get("support_exponent", 0.25)),
        concentration_exponent=float(cfg.get("concentration_exponent", 0.5)),
        max_concentration_ratio=float(cfg.get("max_concentration_ratio", 4.0)),
        min_baseline_effect_size=float(cfg.get("min_baseline_effect_size", 0.25)),
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
        min_precision=float(cfg.get("min_precision", 0.4)),
        min_lift=float(cfg.get("min_lift", 1.5)),
    )
    pareto = _pareto_candidates(scored, cfg)
    selected = _select_binary_diverse_rules(
        [dict(rule, pareto_optimal=True) for rule in pareto],
        centered,
        top_k=int(cfg.get("top_k", 10)),
        max_rule_jaccard=float(cfg.get("max_rule_jaccard", 0.7)),
        min_rule_trajectory_distance=float(cfg.get("min_rule_trajectory_distance", 0.25)),
        max_rule_trajectory_correlation=float(cfg.get("max_rule_trajectory_correlation", 0.995)),
    )

    rules_df = _rules_to_frame(selected, entity_ids)
    rules_df.to_csv(output_dir / "rules.csv", index=False)
    _save_anomaly_scores(entity_ids, anomaly_scores, output_dir / "anomaly_scores.csv")
    pd.DataFrame(
        {
            "entity": entity_ids,
            "trajectory_anomaly_score": anomaly_scores,
            "is_interesting": labels.astype(int),
            "trajectory_cluster": label_info.get("cluster_labels", np.full(len(labels), -1)),
            "cluster_divergence": label_info.get("entity_cluster_divergence", np.full(len(labels), np.nan)),
        }
    ).to_csv(output_dir / "binary_labels.csv", index=False)
    if "cluster_summary" in label_info:
        label_info["cluster_summary"].to_csv(output_dir / "cluster_summary.csv", index=False)
    (output_dir / "rules_natural_language.txt").write_text(
        _natural_language_rules(rules_df),
        encoding="utf-8",
    )
    _save_membership(selected, entity_ids, output_dir / "membership.csv")
    _save_rule_plots(
        selected,
        centered,
        baseline,
        anomaly_scores,
        trajectories.mean(axis=1),
        grid,
        raw_lookup,
        entity_ids,
        output_dir / "plots",
    )
    _save_syflow_distribution_overview(
        selected,
        entity_ids,
        anomaly_scores,
        trajectories,
        output_dir / "subgroup_distribution",
        output_dir / "subgroup_distribution_rules.txt",
    )
    _save_trajectory_pca_distribution(
        selected,
        entity_ids,
        centered,
        output_dir / "trajectory_pca_distribution",
        output_dir / "trajectory_pca_distribution_rules.txt",
    )
    _save_trajectory_signature_distribution(
        selected,
        entity_ids,
        trajectories,
        output_dir / "trajectory_signature_distribution",
        output_dir / "trajectory_signature_distribution_rules.txt",
    )
    _save_fixed_trajectory_metric_distributions(
        selected,
        entity_ids,
        centered,
        baseline,
        trajectories,
        output_dir,
    )
    _save_subgroup_evidence_summary(
        selected,
        entity_ids,
        trajectories,
        output_dir / "subgroup_evidence_summary",
        output_dir / "subgroup_evidence_summary.txt",
    )
    metrics = {
        "n_entities": int(len(entity_ids)),
        "n_interesting_entities": int(labels.sum()),
        "interesting_support": float(labels.mean()),
        "label_strategy": label_info["label_strategy"],
        "anomaly_threshold": label_info.get("anomaly_threshold"),
        "anomaly_quantile": float(cfg.get("anomaly_quantile", 0.9)),
        "selected_cluster_k": label_info.get("selected_cluster_k"),
        "selected_cluster_silhouette": label_info.get("selected_cluster_silhouette"),
        "n_interesting_clusters": label_info.get("n_interesting_clusters"),
        "n_candidate_rules": int(len(candidates)),
        "n_pareto_rules": int(len(pareto)),
        "n_selected_rules": int(len(selected)),
        "grid_size": int(len(grid)),
        "trajectory_centering": "entity_mean",
        "anomaly_score_mean": float(np.mean(anomaly_scores)),
        "anomaly_score_max": float(np.max(anomaly_scores)),
        "min_precision": float(cfg.get("min_precision", 0.4)),
        "min_lift": float(cfg.get("min_lift", 1.5)),
        **forest_metrics,
        "top_quality": float(rules_df["quality"].max()) if not rules_df.empty else None,
        "top_divergence": float(rules_df["divergence"].max()) if not rules_df.empty else None,
    }
    save_json(output_dir / "metrics.json", metrics)
    save_json(
        output_dir / "artifacts.json",
        {
            "rules": "rules.csv",
            "natural_language_rules": "rules_natural_language.txt",
            "binary_labels": "binary_labels.csv",
            "cluster_summary": "cluster_summary.csv",
            "anomaly_scores": "anomaly_scores.csv",
            "forest_summary": "forest_summary.csv",
            "forest_selection": "forest_selection.csv",
            "forest_trees": "forest_trees",
            "forest_tree_manifest": "forest_trees/manifest.csv",
            "forest_performance_questions": "performance_vs_questions.png",
            "subgroup_distribution": "subgroup_distribution.png",
            "subgroup_distribution_overlay": "subgroup_distribution_overlay.png",
            "subgroup_distribution_axis_selection": "subgroup_distribution_axis_selection.csv",
            "subgroup_distribution_rules": "subgroup_distribution_rules.txt",
            "trajectory_pca_distribution": "trajectory_pca_distribution.png",
            "trajectory_pca_distribution_rules": "trajectory_pca_distribution_rules.txt",
            "trajectory_pca_distribution_components": "trajectory_pca_distribution_components.csv",
            "trajectory_signature_distribution": "trajectory_signature_distribution.png",
            "trajectory_signature_distribution_rules": "trajectory_signature_distribution_rules.txt",
            "trajectory_signature_distribution_axis_selection": "trajectory_signature_distribution_axis_selection.csv",
            "trajectory_baseline_deviation_distribution": "trajectory_baseline_deviation_distribution.png",
            "trajectory_baseline_deviation_distribution_rules": "trajectory_baseline_deviation_distribution_rules.txt",
            "trajectory_mean_abs_change_distribution": "trajectory_mean_abs_change_distribution.png",
            "trajectory_mean_abs_change_distribution_rules": "trajectory_mean_abs_change_distribution_rules.txt",
            "subgroup_evidence_summary": "subgroup_evidence_summary.png",
            "subgroup_evidence_summary_key": "subgroup_evidence_summary.txt",
            "membership": "membership.csv",
            "metrics": "metrics.json",
            "plots": "plots",
        },
    )
    return metrics


def _entity_anomaly_scores(centered: np.ndarray, baseline: np.ndarray, metric: str) -> np.ndarray:
    if metric == "wasserstein" and wasserstein_distance is not None:
        return np.asarray([wasserstein_distance(row, baseline) for row in centered], dtype=float)
    return np.linalg.norm(centered - baseline, axis=1)


def _make_binary_trajectory_labels(
    centered: np.ndarray,
    baseline: np.ndarray,
    anomaly_scores: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    strategy = str(cfg.get("label_strategy", "anomaly_quantile"))
    min_positive_entities = int(cfg.get("min_positive_entities", 5))
    if strategy == "clustered_anomaly":
        labels, info = _clustered_anomaly_labels(centered, baseline, anomaly_scores, cfg)
        if int(labels.sum()) >= min_positive_entities:
            return labels, info
        fallback_labels, fallback_info = _quantile_anomaly_labels(
            anomaly_scores, float(cfg.get("anomaly_quantile", 0.9)), min_positive_entities
        )
        fallback_info["label_strategy"] = "clustered_anomaly_fallback_quantile"
        fallback_info.update({k: v for k, v in info.items() if k not in fallback_info})
        return fallback_labels, fallback_info
    return _quantile_anomaly_labels(
        anomaly_scores, float(cfg.get("anomaly_quantile", 0.9)), min_positive_entities
    )


def _quantile_anomaly_labels(
    anomaly_scores: np.ndarray,
    quantile: float,
    min_positive_entities: int,
) -> tuple[np.ndarray, dict]:
    threshold = float(np.quantile(anomaly_scores, quantile))
    labels = anomaly_scores >= threshold
    if int(labels.sum()) < min_positive_entities:
        order = np.argsort(anomaly_scores)[::-1]
        labels = np.zeros(len(anomaly_scores), dtype=bool)
        labels[order[: min(min_positive_entities, len(labels))]] = True
        threshold = float(anomaly_scores[order[min(min_positive_entities, len(labels)) - 1]])
    return labels, {
        "label_strategy": "anomaly_quantile",
        "anomaly_threshold": threshold,
        "cluster_labels": np.full(len(labels), -1),
        "entity_cluster_divergence": np.full(len(labels), np.nan),
    }


def _clustered_anomaly_labels(
    centered: np.ndarray,
    baseline: np.ndarray,
    anomaly_scores: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    k_min = int(cfg.get("cluster_k_min", 2))
    k_max = int(cfg.get("cluster_k_max", 8))
    seed = int(cfg.get("seed", 42))
    n_entities = centered.shape[0]
    k_min = max(2, min(k_min, n_entities - 1))
    k_max = max(k_min, min(k_max, n_entities - 1))
    standardized = centered / np.maximum(np.std(centered, axis=0, ddof=0), 1e-12)
    best = None
    records = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        cluster_labels = model.fit_predict(standardized)
        if len(np.unique(cluster_labels)) < 2:
            continue
        silhouette = float(silhouette_score(standardized, cluster_labels))
        records.append((silhouette, k, cluster_labels))
        if best is None or silhouette > best[0]:
            best = (silhouette, k, cluster_labels)
    if best is None:
        return _quantile_anomaly_labels(
            anomaly_scores, float(cfg.get("anomaly_quantile", 0.9)), int(cfg.get("min_positive_entities", 5))
        )

    silhouette, selected_k, cluster_labels = best
    cluster_rows = []
    cluster_divergence_by_id = {}
    for cluster_id in sorted(np.unique(cluster_labels)):
        mask = cluster_labels == cluster_id
        cluster_mean = centered[mask].mean(axis=0)
        divergence = _aggregate_divergence(centered[mask], baseline, str(cfg.get("divergence_metric", "euclidean")))
        cluster_divergence_by_id[int(cluster_id)] = float(divergence)
        cluster_rows.append(
            {
                "cluster": int(cluster_id),
                "n_entities": int(mask.sum()),
                "support": float(mask.mean()),
                "cluster_divergence": float(divergence),
                "mean_anomaly_score": float(np.mean(anomaly_scores[mask])),
                "median_anomaly_score": float(np.median(anomaly_scores[mask])),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows).sort_values("cluster_divergence", ascending=False)
    min_cluster_support = float(cfg.get("min_cluster_support", cfg.get("min_support", 0.02)))
    eligible = cluster_summary[cluster_summary["support"] >= min_cluster_support].copy()
    if eligible.empty:
        eligible = cluster_summary.copy()
    cluster_quantile = float(cfg.get("cluster_divergence_quantile", 0.75))
    divergence_threshold = float(eligible["cluster_divergence"].quantile(cluster_quantile))
    interesting_clusters = set(
        eligible.loc[eligible["cluster_divergence"] >= divergence_threshold, "cluster"].astype(int).tolist()
    )
    top_n = int(cfg.get("max_interesting_clusters", 2))
    if len(interesting_clusters) > top_n:
        ranked = eligible.sort_values("cluster_divergence", ascending=False)["cluster"].astype(int).tolist()
        interesting_clusters = set(ranked[:top_n])
    labels = np.asarray([int(cluster_id) in interesting_clusters for cluster_id in cluster_labels], dtype=bool)
    entity_cluster_divergence = np.asarray(
        [cluster_divergence_by_id[int(cluster_id)] for cluster_id in cluster_labels], dtype=float
    )
    return labels, {
        "label_strategy": "clustered_anomaly",
        "anomaly_threshold": None,
        "selected_cluster_k": int(selected_k),
        "selected_cluster_silhouette": float(silhouette),
        "n_interesting_clusters": int(len(interesting_clusters)),
        "cluster_labels": cluster_labels.astype(int),
        "entity_cluster_divergence": entity_cluster_divergence,
        "cluster_summary": cluster_summary.assign(
            is_interesting=lambda df: df["cluster"].astype(int).isin(interesting_clusters).astype(int)
        ),
    }


def _build_trajectories(
    table: pd.DataFrame,
    id_col: str,
    target_col: str,
    representation: str,
    grid_size: int,
    min_points: int,
) -> dict:
    years = pd.to_numeric(table["_year"], errors="coerce")
    global_min = float(years.min())
    global_max = float(years.max())
    grid = np.linspace(global_min, global_max, grid_size)
    entity_ids = []
    trajectories = []
    raw_lookup = {}

    for entity, group in table.groupby(id_col, sort=False):
        yy = pd.to_numeric(group[target_col], errors="coerce")
        tt = pd.to_numeric(group["_year"], errors="coerce")
        valid = yy.notna() & tt.notna()
        if int(valid.sum()) < min_points:
            continue
        x = tt[valid].to_numpy(dtype=float)
        y = yy[valid].to_numpy(dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        uniq_x, uniq_idx = np.unique(x, return_index=True)
        x = uniq_x
        y = y[uniq_idx]
        if len(x) < min_points:
            continue
        trajectory = _smooth_or_interp(x, y, grid, representation)
        if np.isnan(trajectory).any():
            continue
        entity_ids.append(entity)
        trajectories.append(trajectory)
        raw_lookup[str(entity)] = {"year": x.tolist(), "target": y.tolist()}

    if not trajectories:
        raise ValueError("DOLPHIN could not build any usable trajectories.")
    return {
        "entity_ids": [str(x) for x in entity_ids],
        "trajectories": np.vstack(trajectories),
        "grid": grid,
        "raw_lookup": raw_lookup,
    }


def _smooth_or_interp(x: np.ndarray, y: np.ndarray, grid: np.ndarray, representation: str) -> np.ndarray:
    base = np.interp(grid, x, y)
    if representation.lower() == "spline" and make_interp_spline is not None and len(x) >= 4:
        try:
            spline = make_interp_spline(x, y, k=min(3, len(x) - 1))
            smoothed = base.copy()
            in_range = (grid >= float(x.min())) & (grid <= float(x.max()))
            smoothed[in_range] = np.asarray(spline(grid[in_range]), dtype=float)
            margin = max(float(np.nanstd(y)), 1e-12)
            return np.clip(smoothed, float(np.nanmin(y) - margin), float(np.nanmax(y) + margin))
        except Exception:
            pass
    return base


def _build_entity_feature_table(table: pd.DataFrame, id_col: str, feature_names: list[str]) -> pd.DataFrame:
    cols = [c for c in feature_names if c in table.columns]
    rows = []
    for entity, group in table.sort_values([id_col, "_year"]).groupby(id_col, sort=False):
        rec = {id_col: str(entity)}
        for col in cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            if values.empty:
                rec[col] = np.nan
            else:
                rec[col] = float(values.iloc[-1])
        rows.append(rec)
    return pd.DataFrame(rows)


def _fit_compact_surrogate_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    centered: np.ndarray,
    baseline: np.ndarray,
    rule_selection_cfg: dict,
    seed: int,
    test_size: float,
    n_search_seeds: int,
    min_trees: int,
    max_trees: int,
    min_questions: int,
    max_questions: int,
    max_depth: int,
    min_samples_leaf: int,
    performance_quantile: float,
    selection_mode: str,
    quality_tolerance: float,
    rule_min_support: float,
    rule_max_support: float,
    support_exponent: float,
    concentration_exponent: float,
    max_concentration_ratio: float,
    output_dir: Path,
) -> tuple[TreeEnsemble, list[int], dict]:
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    X_values = X.to_numpy(dtype=float)
    records = []
    for candidate_seed in range(n_search_seeds):
        candidate = _build_seeded_regression_forest(
            X_values,
            y,
            train_idx,
            test_idx,
            candidate_seed,
            min_trees,
            max_trees,
            min_questions,
            max_questions,
            max_depth,
            min_samples_leaf,
            rule_min_support,
            rule_max_support,
            support_exponent,
            concentration_exponent,
            max_concentration_ratio,
        )
        records.append(candidate["metrics"])

    summary = pd.DataFrame(records)
    best_quality = float(summary["separation_quality"].max())
    if selection_mode == "diversity_tolerance":
        quality_tolerance = min(max(float(quality_tolerance), 0.0), 0.999999)
        threshold = best_quality * (1.0 - quality_tolerance)
        eligible_seeds = summary.loc[summary["separation_quality"] >= threshold, "seed"].astype(int)
        for eligible_seed in eligible_seeds:
            candidate = _build_seeded_regression_forest(
                X_values,
                y,
                train_idx,
                test_idx,
                int(eligible_seed),
                min_trees,
                max_trees,
                min_questions,
                max_questions,
                max_depth,
                min_samples_leaf,
                rule_min_support,
                rule_max_support,
                support_exponent,
                concentration_exponent,
                max_concentration_ratio,
            )
            diversity = _forest_diversity_metrics(
                candidate["trees"],
                X,
                y,
                centered,
                baseline,
                rule_selection_cfg,
            )
            row_mask = summary["seed"] == int(eligible_seed)
            for name, value in diversity.items():
                summary.loc[row_mask, name] = value
        eligible = summary[summary["separation_quality"] >= threshold].sort_values(
            [
                "n_diverse_rules",
                "min_pairwise_trajectory_distance",
                "mean_pairwise_trajectory_distance",
                "total_questions",
                "separation_quality",
                "r2",
                "seed",
            ],
            ascending=[False, False, False, True, False, False, True],
        )
    elif selection_mode == "tolerance":
        quality_tolerance = min(max(float(quality_tolerance), 0.0), 0.999999)
        threshold = best_quality * (1.0 - quality_tolerance)
        eligible = summary[summary["separation_quality"] >= threshold]
        eligible = eligible.sort_values(
            ["total_questions", "separation_quality", "r2", "seed"],
            ascending=[True, False, False, True],
        )
    else:
        threshold = float(summary["separation_quality"].quantile(performance_quantile))
        eligible = summary[summary["separation_quality"] > threshold]
        if eligible.empty:
            eligible = summary[summary["separation_quality"] >= threshold]
    if selection_mode == "quality":
        eligible = eligible.sort_values(
            ["separation_quality", "r2", "total_questions", "seed"],
            ascending=[False, False, True, True],
        )
    elif selection_mode not in {"tolerance", "diversity_tolerance"}:
        eligible = eligible.sort_values(
            ["total_questions", "separation_quality", "seed"],
            ascending=[True, False, True],
        )
    summary.to_csv(output_dir / "forest_summary.csv", index=False)
    eligible.to_csv(output_dir / "forest_selection.csv", index=False)
    chosen = eligible.iloc[0]
    selected_seed = int(chosen["seed"])
    rebuilt = _build_seeded_regression_forest(
        X_values,
        y,
        train_idx,
        test_idx,
        selected_seed,
        min_trees,
        max_trees,
        min_questions,
        max_questions,
        max_depth,
        min_samples_leaf,
        rule_min_support,
        rule_max_support,
        support_exponent,
        concentration_exponent,
        max_concentration_ratio,
    )
    model = TreeEnsemble(rebuilt["trees"])
    selected_indices = list(range(len(model.estimators_)))
    _save_performance_questions_plot(summary, eligible, chosen, threshold, output_dir / "performance_vs_questions")
    metrics = {
        "surrogate_test_r2": float(chosen["r2"]),
        "surrogate_test_mae": float(chosen["mae"]),
        "surrogate_best_candidate_r2": float(summary["r2"].max()),
        "selected_separation_quality": float(chosen["separation_quality"]),
        "best_candidate_separation_quality": float(summary["separation_quality"].max()),
        "forest_performance_quantile": float(performance_quantile),
        "forest_selection_mode": selection_mode,
        "forest_quality_tolerance": (
            float(quality_tolerance)
            if selection_mode in {"tolerance", "diversity_tolerance"}
            else None
        ),
        "surrogate_min_samples_leaf": int(min_samples_leaf),
        "forest_performance_threshold": threshold,
        "n_candidate_forests": int(len(summary)),
        "selected_forest_seed": selected_seed,
        "selected_total_questions": int(chosen["total_questions"]),
        "selected_max_questions_per_tree": int(chosen["max_questions_per_tree"]),
        "n_selected_trees": int(len(selected_indices)),
        "selected_n_diverse_rules": int(chosen.get("n_diverse_rules", 0)),
        "selected_min_pairwise_trajectory_distance": float(
            chosen.get("min_pairwise_trajectory_distance", 0.0)
        ),
        "selected_mean_pairwise_trajectory_distance": float(
            chosen.get("mean_pairwise_trajectory_distance", 0.0)
        ),
        "selected_tree_indices": selected_indices,
        "n_surrogate_train_entities": int(len(train_idx)),
        "n_surrogate_test_entities": int(len(test_idx)),
    }
    return model, selected_indices, metrics


def _build_seeded_regression_forest(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    min_trees: int,
    max_trees: int,
    min_questions: int,
    max_questions: int,
    max_depth: int,
    min_samples_leaf: int,
    rule_min_support: float,
    rule_max_support: float,
    support_exponent: float,
    concentration_exponent: float,
    max_concentration_ratio: float,
) -> dict:
    rng = np.random.RandomState(seed)
    n_trees = int(rng.randint(min_trees, max_trees + 1))
    max_questions_for_seed = int(rng.randint(min_questions, max_questions + 1))
    trees = []
    question_counts = []
    predictions = []
    for _ in range(n_trees):
        bootstrap_idx = rng.choice(train_idx, size=len(train_idx), replace=True)
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            max_leaf_nodes=max_questions_for_seed + 1,
            min_samples_leaf=min_samples_leaf,
            random_state=int(rng.randint(0, 1_000_000)),
        )
        tree.fit(X[bootstrap_idx], y[bootstrap_idx])
        trees.append(tree)
        question_counts.append(_count_tree_questions(tree))
        predictions.append(tree.predict(X[test_idx]))
    forest_prediction = np.vstack(predictions).mean(axis=0)
    separation_quality = _forest_separation_quality(
        trees,
        X[test_idx],
        y[test_idx],
        rule_min_support,
        rule_max_support,
        support_exponent,
        concentration_exponent,
        max_concentration_ratio,
    )
    return {
        "trees": trees,
        "metrics": {
            "seed": seed,
            "r2": float(r2_score(y[test_idx], forest_prediction)),
            "mae": float(mean_absolute_error(y[test_idx], forest_prediction)),
            "separation_quality": float(separation_quality),
            "n_trees": n_trees,
            "max_questions_per_tree": max_questions_for_seed,
            "total_questions": int(sum(question_counts)),
        },
    }


def _fit_binary_surrogate_forest(
    X: pd.DataFrame,
    labels: np.ndarray,
    anomaly_scores: np.ndarray,
    centered: np.ndarray,
    baseline: np.ndarray,
    rule_selection_cfg: dict,
    seed: int,
    test_size: float,
    n_search_seeds: int,
    min_trees: int,
    max_trees: int,
    min_questions: int,
    max_questions: int,
    min_samples_leaf: int,
    quality_tolerance: float,
    accuracy_tolerance: float,
    selection_mode: str,
    rule_min_support: float,
    rule_max_support: float,
    output_dir: Path,
) -> tuple[TreeEnsemble, list[int], dict]:
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels if len(np.unique(labels)) > 1 else None,
    )
    X_values = X.to_numpy(dtype=float)
    records = []
    for candidate_seed in range(n_search_seeds):
        candidate = _build_seeded_binary_forest(
            X_values,
            labels,
            train_idx,
            test_idx,
            candidate_seed,
            min_trees,
            max_trees,
            min_questions,
            max_questions,
            min_samples_leaf,
            rule_min_support,
            rule_max_support,
        )
        records.append(candidate["metrics"])

    summary = pd.DataFrame(records)
    best_quality = float(summary["separation_quality"].max())
    best_accuracy = float(summary["balanced_accuracy"].max())
    quality_tolerance = min(max(float(quality_tolerance), 0.0), 0.999999)
    accuracy_tolerance = min(max(float(accuracy_tolerance), 0.0), 0.999999)
    threshold = best_quality * (1.0 - quality_tolerance)
    accuracy_threshold = max(0.5, best_accuracy * (1.0 - accuracy_tolerance))
    if selection_mode == "separation_quality":
        eligible_mask = summary["separation_quality"] >= threshold
    else:
        eligible_mask = (summary["separation_quality"] >= threshold) & (
            summary["balanced_accuracy"] >= accuracy_threshold
        )
        if not bool(eligible_mask.any()):
            eligible_mask = summary["balanced_accuracy"] >= accuracy_threshold
        if not bool(eligible_mask.any()):
            eligible_mask = summary["separation_quality"] >= threshold
    eligible_seeds = summary.loc[eligible_mask, "seed"].astype(int)
    for eligible_seed in eligible_seeds:
        candidate = _build_seeded_binary_forest(
            X_values,
            labels,
            train_idx,
            test_idx,
            int(eligible_seed),
            min_trees,
            max_trees,
            min_questions,
            max_questions,
            min_samples_leaf,
            rule_min_support,
            rule_max_support,
        )
        diversity = _binary_forest_diversity_metrics(
            candidate["trees"],
            X,
            labels,
            anomaly_scores,
            centered,
            baseline,
            rule_selection_cfg,
        )
        row_mask = summary["seed"] == int(eligible_seed)
        for name, value in diversity.items():
            summary.loc[row_mask, name] = value
    if selection_mode == "separation_quality":
        eligible = summary[eligible_mask].sort_values(
            [
                "separation_quality",
                "n_diverse_rules",
                "min_pairwise_trajectory_distance",
                "mean_pairwise_trajectory_distance",
                "total_questions",
                "balanced_accuracy",
                "seed",
            ],
            ascending=[False, False, False, False, True, False, True],
        )
    else:
        eligible = summary[eligible_mask].sort_values(
            [
                "n_diverse_rules",
                "min_pairwise_trajectory_distance",
                "mean_pairwise_trajectory_distance",
                "balanced_accuracy",
                "f1",
                "total_questions",
                "separation_quality",
                "seed",
            ],
            ascending=[False, False, False, False, False, True, False, True],
        )
    summary.to_csv(output_dir / "forest_summary.csv", index=False)
    eligible.to_csv(output_dir / "forest_selection.csv", index=False)
    chosen = eligible.iloc[0]
    selected_seed = int(chosen["seed"])
    rebuilt = _build_seeded_binary_forest(
        X_values,
        labels,
        train_idx,
        test_idx,
        selected_seed,
        min_trees,
        max_trees,
        min_questions,
        max_questions,
        min_samples_leaf,
        rule_min_support,
        rule_max_support,
    )
    model = TreeEnsemble(rebuilt["trees"])
    selected_indices = list(range(len(model.estimators_)))
    _save_performance_questions_plot(summary, eligible, chosen, threshold, output_dir / "performance_vs_questions")
    metrics = {
        "surrogate_balanced_accuracy": float(chosen["balanced_accuracy"]),
        "surrogate_f1": float(chosen["f1"]),
        "surrogate_best_candidate_balanced_accuracy": float(summary["balanced_accuracy"].max()),
        "selected_separation_quality": float(chosen["separation_quality"]),
        "best_candidate_separation_quality": float(summary["separation_quality"].max()),
        "forest_selection_mode": selection_mode,
        "forest_quality_tolerance": float(quality_tolerance),
        "forest_accuracy_tolerance": float(accuracy_tolerance),
        "surrogate_min_samples_leaf": int(min_samples_leaf),
        "forest_performance_threshold": threshold,
        "forest_accuracy_threshold": accuracy_threshold,
        "n_candidate_forests": int(len(summary)),
        "selected_forest_seed": selected_seed,
        "selected_total_questions": int(chosen["total_questions"]),
        "selected_max_questions_per_tree": int(chosen["max_questions_per_tree"]),
        "n_selected_trees": int(len(selected_indices)),
        "selected_n_diverse_rules": int(chosen.get("n_diverse_rules", 0)),
        "selected_min_pairwise_trajectory_distance": float(chosen.get("min_pairwise_trajectory_distance", 0.0)),
        "selected_mean_pairwise_trajectory_distance": float(chosen.get("mean_pairwise_trajectory_distance", 0.0)),
        "selected_tree_indices": selected_indices,
        "n_surrogate_train_entities": int(len(train_idx)),
        "n_surrogate_test_entities": int(len(test_idx)),
    }
    return model, selected_indices, metrics


def _build_seeded_binary_forest(
    X: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    min_trees: int,
    max_trees: int,
    min_questions: int,
    max_questions: int,
    min_samples_leaf: int,
    rule_min_support: float,
    rule_max_support: float,
) -> dict:
    rng = np.random.RandomState(seed)
    n_trees = int(rng.randint(min_trees, max_trees + 1))
    max_questions_for_seed = int(rng.randint(min_questions, max_questions + 1))
    trees = []
    question_counts = []
    probabilities = []
    y_train = labels[train_idx].astype(int)
    y_test = labels[test_idx].astype(int)
    for _ in range(n_trees):
        bootstrap_idx = rng.choice(train_idx, size=len(train_idx), replace=True)
        tree = DecisionTreeClassifier(
            max_leaf_nodes=max_questions_for_seed + 1,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=int(rng.randint(0, 1_000_000)),
        )
        tree.fit(X[bootstrap_idx], labels[bootstrap_idx].astype(int))
        trees.append(tree)
        question_counts.append(_count_tree_questions(tree))
        if len(tree.classes_) == 1:
            cls = int(tree.classes_[0])
            probabilities.append(np.full(len(test_idx), float(cls)))
        else:
            positive_index = int(np.where(tree.classes_ == 1)[0][0])
            probabilities.append(tree.predict_proba(X[test_idx])[:, positive_index])
    mean_prob = np.vstack(probabilities).mean(axis=0)
    pred = mean_prob >= 0.5
    separation_quality = _binary_forest_separation_quality(
        trees,
        X[test_idx],
        y_test.astype(bool),
        rule_min_support,
        rule_max_support,
    )
    return {
        "trees": trees,
        "metrics": {
            "seed": seed,
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred.astype(int))),
            "f1": float(f1_score(y_test, pred.astype(int), zero_division=0)),
            "separation_quality": float(separation_quality),
            "n_trees": n_trees,
            "max_questions_per_tree": max_questions_for_seed,
            "total_questions": int(sum(question_counts)),
            "train_positive_rate": float(y_train.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
    }


def _forest_separation_quality(
    trees: list[DecisionTreeRegressor],
    X: np.ndarray,
    scores: np.ndarray,
    min_support: float,
    max_support: float,
    support_exponent: float,
    concentration_exponent: float,
    max_concentration_ratio: float,
) -> float:
    qualities = []
    global_iqr = max(_iqr(scores), 1e-12)
    for tree in trees:
        leaf_ids = tree.apply(X)
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            support = float(mask.mean())
            if support < min_support or support > max_support or mask.sum() < 3 or (~mask).sum() < 3:
                continue
            subgroup = scores[mask]
            complement = scores[~mask]
            divergence = _wasserstein(subgroup, complement)
            concentration = min(
                max_concentration_ratio,
                (_iqr(complement) + 0.05 * global_iqr) / (_iqr(subgroup) + 0.05 * global_iqr),
            )
            qualities.append((support**support_exponent) * divergence * (concentration**concentration_exponent))
    if not qualities:
        return 0.0
    return float(np.mean(sorted(qualities, reverse=True)[:3]))


def _forest_diversity_metrics(
    trees: list[DecisionTreeRegressor],
    X: pd.DataFrame,
    anomaly_scores: np.ndarray,
    centered: np.ndarray,
    baseline: np.ndarray,
    cfg: dict,
) -> dict:
    model = TreeEnsemble(trees)
    candidates = _surrogate_rules_from_forest(
        model,
        list(range(len(trees))),
        X,
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
    )
    scored = _score_candidate_rules(
        candidates,
        anomaly_scores,
        centered,
        baseline,
        divergence_metric=str(cfg.get("divergence_metric", "euclidean")),
        support_exponent=float(cfg.get("support_exponent", 0.25)),
        concentration_exponent=float(cfg.get("concentration_exponent", 0.5)),
        max_concentration_ratio=float(cfg.get("max_concentration_ratio", 4.0)),
        min_baseline_effect_size=float(cfg.get("min_baseline_effect_size", 0.25)),
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
        evaluation_anomaly_quantile=float(cfg.get("evaluation_anomaly_quantile", 0.9)),
    )
    pareto = _pareto_candidates(scored, cfg)
    selected = _weighted_covering(
        [dict(rule, pareto_optimal=True) for rule in pareto],
        anomaly_scores,
        centered,
        baseline,
        top_k=int(cfg.get("top_k", 10)),
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
        coverage_decay=float(cfg.get("coverage_decay", 0.1)),
        support_exponent=float(cfg.get("support_exponent", 0.25)),
        concentration_exponent=float(cfg.get("concentration_exponent", 0.5)),
        max_concentration_ratio=float(cfg.get("max_concentration_ratio", 4.0)),
        divergence_metric=str(cfg.get("divergence_metric", "euclidean")),
        max_rule_jaccard=float(cfg.get("max_rule_jaccard", 0.7)),
        min_rule_trajectory_distance=float(cfg.get("min_rule_trajectory_distance", 0.25)),
        max_rule_trajectory_correlation=float(cfg.get("max_rule_trajectory_correlation", 0.995)),
    )
    pairwise_distances = [
        _standardized_trajectory_distance(left["trajectory_mean"], right["trajectory_mean"], centered)
        for i, left in enumerate(selected)
        for right in selected[i + 1 :]
    ]
    return {
        "n_candidate_rules": int(len(candidates)),
        "n_pareto_rules": int(len(pareto)),
        "n_diverse_rules": int(len(selected)),
        "min_pairwise_trajectory_distance": (
            float(min(pairwise_distances)) if pairwise_distances else 0.0
        ),
        "mean_pairwise_trajectory_distance": (
            float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
        ),
    }


def _binary_forest_separation_quality(
    trees: list[DecisionTreeClassifier],
    X: np.ndarray,
    labels: np.ndarray,
    min_support: float,
    max_support: float,
) -> float:
    global_rate = max(float(labels.mean()), 1e-12)
    qualities = []
    for tree in trees:
        leaf_ids = tree.apply(X)
        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            support = float(mask.mean())
            if support < min_support or support > max_support or mask.sum() < 3:
                continue
            precision = float(labels[mask].mean())
            recall = float(labels[mask].sum() / max(labels.sum(), 1))
            lift = precision / global_rate
            qualities.append((support**0.25) * precision * np.sqrt(recall) * min(lift, 5.0))
    if not qualities:
        return 0.0
    return float(np.mean(sorted(qualities, reverse=True)[:3]))


def _binary_forest_diversity_metrics(
    trees: list[DecisionTreeClassifier],
    X: pd.DataFrame,
    labels: np.ndarray,
    anomaly_scores: np.ndarray,
    centered: np.ndarray,
    baseline: np.ndarray,
    cfg: dict,
) -> dict:
    model = TreeEnsemble(trees)
    candidates = _surrogate_rules_from_forest(
        model,
        list(range(len(trees))),
        X,
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
    )
    scored = _score_binary_candidate_rules(
        candidates,
        labels,
        anomaly_scores,
        centered,
        baseline,
        divergence_metric=str(cfg.get("divergence_metric", "euclidean")),
        support_exponent=float(cfg.get("support_exponent", 0.25)),
        concentration_exponent=float(cfg.get("concentration_exponent", 0.5)),
        max_concentration_ratio=float(cfg.get("max_concentration_ratio", 4.0)),
        min_baseline_effect_size=float(cfg.get("min_baseline_effect_size", 0.25)),
        min_support=float(cfg.get("min_support", 0.05)),
        max_support=float(cfg.get("max_support", 0.6)),
        min_precision=float(cfg.get("min_precision", 0.4)),
        min_lift=float(cfg.get("min_lift", 1.5)),
    )
    pareto = _pareto_candidates(scored, cfg)
    selected = _select_binary_diverse_rules(
        [dict(rule, pareto_optimal=True) for rule in pareto],
        centered,
        top_k=int(cfg.get("top_k", 10)),
        max_rule_jaccard=float(cfg.get("max_rule_jaccard", 0.7)),
        min_rule_trajectory_distance=float(cfg.get("min_rule_trajectory_distance", 0.25)),
        max_rule_trajectory_correlation=float(cfg.get("max_rule_trajectory_correlation", 0.995)),
    )
    pairwise_distances = [
        _standardized_trajectory_distance(left["trajectory_mean"], right["trajectory_mean"], centered)
        for i, left in enumerate(selected)
        for right in selected[i + 1 :]
    ]
    return {
        "n_candidate_rules": int(len(candidates)),
        "n_pareto_rules": int(len(pareto)),
        "n_diverse_rules": int(len(selected)),
        "min_pairwise_trajectory_distance": float(min(pairwise_distances)) if pairwise_distances else 0.0,
        "mean_pairwise_trajectory_distance": float(np.mean(pairwise_distances)) if pairwise_distances else 0.0,
    }


def _count_tree_questions(tree: DecisionTreeRegressor) -> int:
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right

    def visit(node: int) -> int:
        if children_left[node] == children_right[node]:
            return 0
        return 1 + visit(int(children_left[node])) + visit(int(children_right[node]))

    return int(visit(0))


def _save_performance_questions_plot(
    summary: pd.DataFrame,
    eligible: pd.DataFrame,
    chosen: pd.Series,
    threshold: float,
    output_base: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(summary["total_questions"], summary["separation_quality"], s=35, alpha=0.45, color="#4C78A8")
    ax.scatter(
        eligible["total_questions"],
        eligible["separation_quality"],
        s=50,
        alpha=0.8,
        color="#F2A541",
        label="Within selection tolerance",
    )
    ax.axhline(threshold, color="#777777", linestyle="--", linewidth=1.5, label="Eligibility threshold")
    ax.scatter([chosen["total_questions"]], [chosen["separation_quality"]], s=120, color="#D62728", label="Selected forest")
    ax.set_xlabel("Total Number of Questions in Forest", fontsize=12)
    ax.set_ylabel("Held-out Subgroup Separation Quality", fontsize=12)
    ax.set_title("DOLPHIN Separation Quality vs Total Questions", fontsize=14)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_surrogate_forest_trees(
    forest: TreeEnsemble,
    selected_indices: list[int],
    feature_names: list[str],
    X: np.ndarray,
    entity_ids: list[str],
    anomaly_scores: np.ndarray,
    centered: np.ndarray,
    trajectories: np.ndarray,
    target_col: str,
    high_anomaly_percentile: float,
    min_baseline_effect_size: float,
    output_dir: Path,
) -> None:
    n_trees = len(selected_indices)
    if n_trees == 0:
        return
    output_dir = ensure_dir(output_dir)
    manifest = []
    for rank, tree_index in enumerate(selected_indices, start=1):
        tree = forest.estimators_[tree_index]
        leaf_summary = _dolphin_leaf_summary(
            tree,
            X,
            entity_ids,
            anomaly_scores,
            centered,
            trajectories,
            high_anomaly_percentile,
            min_baseline_effect_size,
        )
        base = output_dir / f"tree_{rank:02d}_idx_{tree_index:03d}"
        save_graphviz_style_tree(
            tree,
            feature_names,
            leaf_label=lambda node, summary=leaf_summary: _dolphin_leaf_label(summary[node], target_col),
            leaf_color=lambda node, summary=leaf_summary: "#f3a6a0" if summary[node]["high_anomaly"] else "#b9d8ee",
            title=f"DOLPHIN surrogate tree {rank} (forest index {tree_index})",
            output_base=base,
            fontsize=18,
        )
        manifest.append(
            {
                "tree_rank": rank,
                "forest_index": int(tree_index),
                "depth": int(tree.tree_.max_depth),
                "leaves": int(tree.tree_.n_leaves),
                "png": base.name + ".png",
                "pdf": base.name + ".pdf",
                "svg": base.name + ".svg",
            }
        )
    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)


def _dolphin_leaf_label(row: dict, target_col: str) -> str:
    status = "HIGH ANOMALY" if row.get("high_anomaly") else "LOW/MID ANOMALY"
    change = row.get("target_change", np.nan)
    global_change = row.get("global_target_change", np.nan)
    label = (
        status
        + f"\nn={row.get('n_entities', 0)} ({100 * row.get('support', 0):.1f}%)"
        + f"\neffect={row.get('baseline_effect_size', np.nan):.2f} SD"
        + f"\ngap={change - global_change:+.3g}"
    )
    return label


def _dolphin_leaf_summary(
    tree: DecisionTreeRegressor,
    X: np.ndarray,
    entity_ids: list[str],
    anomaly_scores: np.ndarray,
    centered: np.ndarray,
    trajectories: np.ndarray,
    high_anomaly_percentile: float,
    min_baseline_effect_size: float,
) -> dict[int, dict]:
    leaf_ids = tree.apply(X)
    baseline = centered.mean(axis=0)
    anomaly_percentiles = pd.Series(anomaly_scores).rank(method="average", pct=True).to_numpy(dtype=float)
    global_change = float(np.mean(trajectories[:, -1] - trajectories[:, 0]))
    result = {}
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        subgroup_mean = centered[mask].mean(axis=0)
        effect = _standardized_trajectory_distance(subgroup_mean, baseline, centered)
        mean_percentile = float(np.mean(anomaly_percentiles[mask]))
        target_change = float(np.mean(trajectories[mask, -1] - trajectories[mask, 0]))
        members = [str(entity_ids[i]) for i in np.flatnonzero(mask)]
        preview = ", ".join(members[:6])
        if len(members) > 6:
            preview += f", +{len(members) - 6} more"
        result[int(leaf_id)] = {
            "n_entities": int(mask.sum()),
            "support": float(mask.mean()),
            "mean_anomaly": float(np.mean(anomaly_scores[mask])),
            "mean_anomaly_percentile": mean_percentile,
            "baseline_effect_size": float(effect),
            "target_change": target_change,
            "global_target_change": global_change,
            "entity_preview": preview,
            "high_anomaly": bool(
                mean_percentile >= high_anomaly_percentile and effect >= min_baseline_effect_size
            ),
        }
    return result


def _rewrite_dolphin_tree_labels(
    artists,
    tree: DecisionTreeRegressor,
    feature_names: list[str],
    leaf_summary: dict[int, dict],
    target_col: str,
) -> None:
    tree_ = tree.tree_
    target_label = _pretty_feature_name(target_col)
    for artist in artists:
        text = artist.get_text()
        if text in {"True", "False"}:
            artist.set_text("")
            continue
        node_id = _tree_artist_node_id(text)
        if node_id is None or node_id >= tree_.node_count:
            continue
        if tree_.feature[node_id] != _tree.TREE_UNDEFINED:
            feature = feature_names[int(tree_.feature[node_id])]
            artist.set_text(f"{feature} <= {float(tree_.threshold[node_id]):.4g}")
            continue
        row = leaf_summary.get(node_id, {})
        status = "HIGH TRAJECTORY ANOMALY" if row.get("high_anomaly") else "NOT HIGH ANOMALY"
        change = row.get("target_change", np.nan)
        global_change = row.get("global_target_change", np.nan)
        relative = change - global_change
        artist.set_text(
            "Leaf: " + status
            + f"\nn={row.get('n_entities', 0)} ({100 * row.get('support', 0):.1f}%)"
            + f"\nmean anomaly={row.get('mean_anomaly', np.nan):.3g}"
            + f" (P{100 * row.get('mean_anomaly_percentile', np.nan):.0f})"
            + f"\ntrajectory effect={row.get('baseline_effect_size', np.nan):.2f} SD"
            + f"\n{target_label} change={change:+.3g}"
            + f"\nglobal change={global_change:+.3g}"
            + f"\ndifference={relative:+.3g}"
        )


def _tree_artist_node_id(text: str) -> int | None:
    for token in text.replace("\n", " ").split():
        if token.startswith("#"):
            try:
                return int(token[1:])
            except ValueError:
                return None
    return None


def _surrogate_rules_from_forest(
    model: TreeEnsemble,
    selected_tree_indices: list[int],
    X: pd.DataFrame,
    min_support: float,
    max_support: float,
) -> list[dict]:

    seen = set()
    candidates = []
    feature_names = list(X.columns)
    for tree_idx in selected_tree_indices:
        estimator = model.estimators_[tree_idx]
        tree = estimator.tree_

        def visit(node: int, path: list[Selector]) -> None:
            if path:
                selectors = _simplify_selectors(path)
                if selectors:
                    key = _rule_key(selectors)
                    if key not in seen:
                        mask = _apply_selectors(selectors, X)
                        support = float(mask.mean())
                        if min_support <= support <= max_support and int(mask.sum()) >= 3:
                            seen.add(key)
                            candidates.append(
                                {
                                    "selectors": selectors,
                                    "mask": mask,
                                    "source_tree": int(tree_idx),
                                    "source_leaf": int(node),
                                }
                            )
            left = tree.children_left[node]
            right = tree.children_right[node]
            if left == right:
                return

            feature = feature_names[int(tree.feature[node])]
            threshold = float(tree.threshold[node])
            visit(left, path + [Selector(feature, "<=", threshold)])
            visit(right, path + [Selector(feature, ">", threshold)])

        visit(0, [])

    return candidates


def _simplify_selectors(selectors: list[Selector]) -> list[Selector]:
    lower: dict[str, float] = {}
    upper: dict[str, float] = {}
    for selector in selectors:
        if selector.op == ">":
            lower[selector.feature] = max(lower.get(selector.feature, -np.inf), selector.threshold)
        elif selector.op == "<=":
            upper[selector.feature] = min(upper.get(selector.feature, np.inf), selector.threshold)

    simplified = []
    for feature in sorted(set(lower) | set(upper)):
        if lower.get(feature, -np.inf) >= upper.get(feature, np.inf):
            return []
        if feature in lower:
            simplified.append(Selector(feature, ">", lower[feature]))
        if feature in upper:
            simplified.append(Selector(feature, "<=", upper[feature]))
    return simplified


def _apply_selectors(selectors: list[Selector], X: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(X), dtype=bool)
    for selector in selectors:
        mask &= selector.apply(X)
    return mask


def _score_candidate_rules(
    candidates: list[dict],
    anomaly_scores: np.ndarray,
    centered: np.ndarray,
    baseline: np.ndarray,
    divergence_metric: str,
    support_exponent: float,
    concentration_exponent: float,
    max_concentration_ratio: float,
    min_baseline_effect_size: float,
    min_support: float,
    max_support: float,
    evaluation_anomaly_quantile: float,
) -> list[dict]:
    scored = []
    n_total = len(anomaly_scores)
    interesting_labels = anomaly_scores >= float(np.quantile(anomaly_scores, evaluation_anomaly_quantile))
    total_interesting = max(int(interesting_labels.sum()), 1)
    global_interesting_rate = max(float(interesting_labels.mean()), 1e-12)
    for candidate in candidates:
        mask = candidate["mask"]
        support = float(mask.mean())
        n_entities = int(mask.sum())
        if support < min_support or support > max_support or n_entities < 3:
            continue
        stats = _distribution_separation(anomaly_scores, mask, max_concentration_ratio)
        trajectory_mean = np.mean(centered[mask], axis=0)
        trajectory_divergence = _aggregate_divergence(centered[mask], baseline, divergence_metric)
        baseline_effect_size = _standardized_trajectory_distance(trajectory_mean, baseline, centered)
        if baseline_effect_size < min_baseline_effect_size:
            continue
        precision = float(interesting_labels[mask].mean())
        recall = float(interesting_labels[mask].sum() / total_interesting)
        lift = float(precision / global_interesting_rate)
        base_quality = (support**support_exponent) * trajectory_divergence
        quality = base_quality * (stats["concentration_ratio"]**concentration_exponent)
        trajectory_stats = _trajectory_diagnostics(trajectory_mean, baseline)
        scored.append(
            {
                **candidate,
                "n_entities": n_entities,
                "support": support,
                **stats,
                "anomaly_distribution_divergence": float(stats["divergence"]),
                "divergence": float(trajectory_divergence),
                "baseline_effect_size": float(baseline_effect_size),
                "trajectory_mean": trajectory_mean,
                **trajectory_stats,
                "base_quality": float(base_quality),
                "precision": precision,
                "recall": recall,
                "lift": lift,
                "n_interesting_entities": int(interesting_labels[mask].sum()),
                "quality": float(quality),
                "weighted_quality": float(quality),
                "effective_n_entities": float(n_entities),
                "complexity": int(len(candidate["selectors"])),
                "population_size": int(n_total),
            }
        )
    return sorted(scored, key=lambda r: r["quality"], reverse=True)


def _score_binary_candidate_rules(
    candidates: list[dict],
    labels: np.ndarray,
    anomaly_scores: np.ndarray,
    centered: np.ndarray,
    baseline: np.ndarray,
    divergence_metric: str,
    support_exponent: float,
    concentration_exponent: float,
    max_concentration_ratio: float,
    min_baseline_effect_size: float,
    min_support: float,
    max_support: float,
    min_precision: float,
    min_lift: float,
) -> list[dict]:
    scored = []
    n_total = len(anomaly_scores)
    total_positive = max(int(labels.sum()), 1)
    global_rate = max(float(labels.mean()), 1e-12)
    for candidate in candidates:
        mask = candidate["mask"]
        support = float(mask.mean())
        n_entities = int(mask.sum())
        if support < min_support or support > max_support or n_entities < 3:
            continue
        precision = float(labels[mask].mean())
        recall = float(labels[mask].sum() / total_positive)
        lift = float(precision / global_rate)
        if precision < min_precision or lift < min_lift:
            continue
        stats = _distribution_separation(anomaly_scores, mask, max_concentration_ratio)
        trajectory_mean = np.mean(centered[mask], axis=0)
        trajectory_divergence = _aggregate_divergence(centered[mask], baseline, divergence_metric)
        baseline_effect_size = _standardized_trajectory_distance(trajectory_mean, baseline, centered)
        if baseline_effect_size < min_baseline_effect_size:
            continue
        base_quality = (support**support_exponent) * trajectory_divergence
        trajectory_quality = base_quality * (stats["concentration_ratio"] ** concentration_exponent)
        label_quality = precision * np.sqrt(max(recall, 0.0)) * min(lift, 5.0)
        quality = trajectory_quality * label_quality
        trajectory_stats = _trajectory_diagnostics(trajectory_mean, baseline)
        scored.append(
            {
                **candidate,
                "n_entities": n_entities,
                "support": support,
                **stats,
                "anomaly_distribution_divergence": float(stats["divergence"]),
                "divergence": float(trajectory_divergence),
                "baseline_effect_size": float(baseline_effect_size),
                "trajectory_mean": trajectory_mean,
                **trajectory_stats,
                "base_quality": float(base_quality),
                "trajectory_quality": float(trajectory_quality),
                "label_quality": float(label_quality),
                "precision": precision,
                "recall": recall,
                "lift": lift,
                "n_interesting_entities": int(labels[mask].sum()),
                "quality": float(quality),
                "weighted_quality": float(quality),
                "effective_n_entities": float(n_entities),
                "complexity": int(len(candidate["selectors"])),
                "population_size": int(n_total),
            }
        )
    return sorted(scored, key=lambda r: r["quality"], reverse=True)


def _select_binary_diverse_rules(
    rules: list[dict],
    centered: np.ndarray,
    top_k: int,
    max_rule_jaccard: float,
    min_rule_trajectory_distance: float,
    max_rule_trajectory_correlation: float,
) -> list[dict]:
    selected = []
    for rule in sorted(rules, key=lambda r: r["quality"], reverse=True):
        mask = rule["mask"]
        if any(_mask_jaccard(mask, prior["mask"]) > max_rule_jaccard for prior in selected):
            continue
        trajectory_mean = rule.get("trajectory_mean")
        if trajectory_mean is None:
            trajectory_mean = np.mean(centered[mask], axis=0)
        if any(
            _standardized_trajectory_distance(trajectory_mean, prior["trajectory_mean"], centered)
            < min_rule_trajectory_distance
            for prior in selected
        ):
            continue
        if any(
            abs(_trajectory_correlation(trajectory_mean, prior["trajectory_mean"]))
            > max_rule_trajectory_correlation
            for prior in selected
        ):
            continue
        selected.append(dict(rule, trajectory_mean=trajectory_mean))
        if len(selected) >= top_k:
            break
    return selected


def _distribution_separation(values: np.ndarray, mask: np.ndarray, max_concentration_ratio: float) -> dict:
    subgroup = np.asarray(values[mask], dtype=float)
    complement = np.asarray(values[~mask], dtype=float)
    divergence = _wasserstein(subgroup, complement)
    subgroup_iqr = _iqr(subgroup)
    complement_iqr = _iqr(complement)
    scale = max(_iqr(np.asarray(values, dtype=float)), 1e-12)
    concentration_ratio = min(max_concentration_ratio, (complement_iqr + 0.05 * scale) / (subgroup_iqr + 0.05 * scale))
    median_shift = float(np.median(subgroup) - np.median(complement))
    if abs(median_shift) <= 0.1 * scale:
        location = "shape/dispersion shifted"
    elif median_shift > 0:
        location = "upper shifted"
    else:
        location = "lower shifted"
    return {
        "divergence": float(divergence),
        "subgroup_iqr": float(subgroup_iqr),
        "complement_iqr": float(complement_iqr),
        "concentration_ratio": float(concentration_ratio),
        "median_shift": median_shift,
        "distribution_location": location,
    }


def _trajectory_diagnostics(trajectory_mean: np.ndarray, baseline: np.ndarray) -> dict:
    trajectory = np.asarray(trajectory_mean, dtype=float)
    base = np.asarray(baseline, dtype=float)
    subgroup_change = float(trajectory[-1] - trajectory[0])
    baseline_change = float(base[-1] - base[0])
    change_difference = subgroup_change - baseline_change
    peak_gap_index = int(np.argmax(np.abs(trajectory - base)))
    return {
        "subgroup_trend_start": float(trajectory[0]),
        "subgroup_trend_end": float(trajectory[-1]),
        "subgroup_trend_change": subgroup_change,
        "baseline_trend_start": float(base[0]),
        "baseline_trend_end": float(base[-1]),
        "baseline_trend_change": baseline_change,
        "trend_change_difference": float(change_difference),
        "trajectory_peak_gap": float(trajectory[peak_gap_index] - base[peak_gap_index]),
        "trajectory_peak_gap_index": peak_gap_index,
        "subgroup_trend_direction": _direction_label(subgroup_change),
        "relative_trend_direction": _relative_direction_label(change_difference),
    }


def _direction_label(value: float) -> str:
    if abs(value) <= 1e-9:
        return "flat"
    return "increasing" if value > 0 else "decreasing"


def _relative_direction_label(value: float) -> str:
    if abs(value) <= 1e-9:
        return "similar start-to-end change"
    return "stronger increase than baseline" if value > 0 else "weaker increase than baseline"


def _wasserstein(left: np.ndarray, right: np.ndarray, left_weights=None, right_weights=None) -> float:
    if wasserstein_distance is not None:
        return float(wasserstein_distance(left, right, u_weights=left_weights, v_weights=right_weights))
    return float(abs(np.average(left, weights=left_weights) - np.average(right, weights=right_weights)))


def _iqr(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def _aggregate_divergence(subgroup_centered: np.ndarray, baseline: np.ndarray, metric: str) -> float:
    subgroup_mean = np.average(subgroup_centered, axis=0)
    if metric == "wasserstein" and wasserstein_distance is not None:
        return float(wasserstein_distance(subgroup_mean, baseline))
    return float(np.linalg.norm(subgroup_mean - baseline))


def _weighted_aggregate_divergence(
    subgroup_centered: np.ndarray,
    subgroup_weights: np.ndarray,
    baseline: np.ndarray,
    metric: str,
) -> float:
    if float(subgroup_weights.sum()) <= 0:
        return 0.0
    subgroup_mean = np.average(subgroup_centered, axis=0, weights=subgroup_weights)
    if metric == "wasserstein" and wasserstein_distance is not None:
        return float(wasserstein_distance(subgroup_mean, baseline))
    return float(np.linalg.norm(subgroup_mean - baseline))


def _pareto_candidates(rules: list[dict], cfg: dict) -> list[dict]:
    if not cfg.get("pareto_filter", True):
        return sorted(rules, key=lambda r: r["quality"], reverse=True)
    return _pareto_filter(rules, quality_tolerance=float(cfg.get("pareto_quality_tolerance", 0.0)))


def _pareto_filter(rules: list[dict], quality_tolerance: float = 0.0) -> list[dict]:
    kept = []
    tolerance = max(float(quality_tolerance), 0.0)
    for rule in rules:
        dominated = any(
            other is not rule
            and other["complexity"] < rule["complexity"]
            and other["quality"] >= rule["quality"] * (1.0 + tolerance)
            for other in rules
        )
        if not dominated:
            kept.append(rule)
    return sorted(kept, key=lambda r: r["quality"], reverse=True)


def _weighted_covering(
    rules: list[dict],
    anomaly_scores: np.ndarray,
    centered: np.ndarray,
    baseline: np.ndarray,
    top_k: int,
    min_support: float,
    max_support: float,
    coverage_decay: float,
    support_exponent: float,
    concentration_exponent: float,
    max_concentration_ratio: float,
    divergence_metric: str,
    max_rule_jaccard: float,
    min_rule_trajectory_distance: float,
    max_rule_trajectory_correlation: float,
) -> list[dict]:
    remaining = [dict(rule) for rule in rules]
    weights = np.ones(len(anomaly_scores), dtype=float)
    selected = []
    min_effective_n = max(3.0, min_support * len(anomaly_scores))
    max_effective_n = max_support * len(anomaly_scores)

    while remaining and len(selected) < top_k:
        rescored = []
        for rule in remaining:
            mask = rule["mask"]
            if any(_mask_jaccard(mask, prior["mask"]) > max_rule_jaccard for prior in selected):
                continue
            trajectory_mean = np.mean(centered[mask], axis=0)
            if any(
                _standardized_trajectory_distance(trajectory_mean, prior["trajectory_mean"], centered)
                < min_rule_trajectory_distance
                for prior in selected
            ):
                continue
            if any(
                abs(_trajectory_correlation(trajectory_mean, prior["trajectory_mean"]))
                > max_rule_trajectory_correlation
                for prior in selected
            ):
                continue
            local_weights = weights[mask]
            effective_n = float(local_weights.sum())
            if effective_n < min_effective_n or effective_n > max_effective_n:
                continue
            complement_weights = weights[~mask]
            if float(complement_weights.sum()) <= 0:
                continue
            anomaly_distribution_divergence = _wasserstein(
                anomaly_scores[mask],
                anomaly_scores[~mask],
                local_weights,
                complement_weights,
            )
            divergence = _weighted_aggregate_divergence(
                centered[mask], local_weights, baseline, divergence_metric
            )
            subgroup_iqr = _weighted_iqr(anomaly_scores[mask], local_weights)
            complement_iqr = _weighted_iqr(anomaly_scores[~mask], complement_weights)
            global_scale = max(_weighted_iqr(anomaly_scores, weights), 1e-12)
            concentration_ratio = min(
                max_concentration_ratio,
                (complement_iqr + 0.05 * global_scale) / (subgroup_iqr + 0.05 * global_scale),
            )
            effective_support = effective_n / float(weights.sum())
            base_quality = (effective_support**support_exponent) * divergence
            weighted_quality = base_quality * (concentration_ratio**concentration_exponent)
            median_shift = _weighted_median(anomaly_scores[mask], local_weights) - _weighted_median(
                anomaly_scores[~mask], complement_weights
            )
            updated = dict(rule)
            updated["weighted_quality"] = float(weighted_quality)
            updated["effective_n_entities"] = float(effective_n)
            updated["divergence"] = float(divergence)
            updated["anomaly_distribution_divergence"] = float(anomaly_distribution_divergence)
            updated["trajectory_mean"] = trajectory_mean
            updated["base_quality"] = float(base_quality)
            updated["subgroup_iqr"] = float(subgroup_iqr)
            updated["complement_iqr"] = float(complement_iqr)
            updated["concentration_ratio"] = float(concentration_ratio)
            updated["median_shift"] = float(median_shift)
            updated["distribution_location"] = _location_label(median_shift, global_scale)
            updated["quality"] = float(weighted_quality)
            updated.update(_trajectory_diagnostics(trajectory_mean, baseline))
            rescored.append(updated)

        if not rescored:
            break

        best = max(rescored, key=lambda r: r["weighted_quality"])
        selected.append(best)
        weights[best["mask"]] *= coverage_decay
        selected_key = _rule_key(best["selectors"])
        remaining = [r for r in remaining if _rule_key(r["selectors"]) != selected_key]

    return selected


def _standardized_trajectory_distance(left: np.ndarray, right: np.ndarray, population: np.ndarray) -> float:
    scale = np.std(population, axis=0)
    positive = scale[scale > 1e-12]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    scale = np.where(scale > 1e-12, scale, fallback)
    return float(np.sqrt(np.mean(((np.asarray(left) - np.asarray(right)) / scale) ** 2)))


def _trajectory_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 1.0 if np.allclose(left, right) else 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _weighted_iqr(values: np.ndarray, weights: np.ndarray) -> float:
    return float(_weighted_quantile(values, weights, 0.75) - _weighted_quantile(values, weights, 0.25))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    return float(_weighted_quantile(values, weights, 0.5))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = np.asarray(values)[order]
    ordered_weights = np.asarray(weights)[order]
    cumulative = np.cumsum(ordered_weights)
    cutoff = quantile * cumulative[-1]
    return float(ordered_values[min(int(np.searchsorted(cumulative, cutoff, side="left")), len(ordered_values) - 1)])


def _location_label(median_shift: float, scale: float) -> str:
    if abs(median_shift) <= 0.1 * scale:
        return "shape/dispersion shifted"
    return "upper shifted" if median_shift > 0 else "lower shifted"


def _mask_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _make_selectors(
    X: pd.DataFrame,
    n_bins: int,
    min_support: float,
    max_support: float,
) -> list[Selector]:
    selectors = []
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    for col in X.columns:
        values = pd.to_numeric(X[col], errors="coerce")
        if values.nunique(dropna=True) < 3:
            continue
        for threshold in sorted(set(values.quantile(quantiles).dropna().astype(float))):
            for op in ["<=", ">"]:
                selector = Selector(col, op, float(threshold))
                support = selector.apply(X).mean()
                if min_support <= support <= max_support:
                    selectors.append(selector)
    return selectors


def _beam_search_rules(
    X: pd.DataFrame,
    trajectories: np.ndarray,
    selectors: list[Selector],
    beam_width: int,
    max_depth: int,
    alpha: float,
    min_support: float,
    max_support: float,
) -> list[dict]:
    seen = set()
    all_rules = []
    beam = []

    for selector in selectors:
        mask = selector.apply(X)
        rule = _score_rule([selector], mask, trajectories, alpha, min_support, max_support)
        if rule is None:
            continue
        key = _rule_key(rule["selectors"])
        seen.add(key)
        beam.append(rule)
        all_rules.append(rule)
    beam = sorted(beam, key=lambda r: r["quality"], reverse=True)[:beam_width]

    for _depth in range(2, max_depth + 1):
        next_beam = []
        for rule in beam:
            used_features = {s.feature for s in rule["selectors"]}
            for selector in selectors:
                if selector.feature in used_features:
                    continue
                selectors_new = list(rule["selectors"]) + [selector]
                key = _rule_key(selectors_new)
                if key in seen:
                    continue
                seen.add(key)
                mask = rule["mask"] & selector.apply(X)
                scored = _score_rule(selectors_new, mask, trajectories, alpha, min_support, max_support)
                if scored is None:
                    continue
                next_beam.append(scored)
                all_rules.append(scored)
        beam = sorted(next_beam, key=lambda r: r["quality"], reverse=True)[:beam_width]
        if not beam:
            break
    return sorted(all_rules, key=lambda r: r["quality"], reverse=True)


def _score_rule(
    selectors: list[Selector],
    mask: np.ndarray,
    trajectories: np.ndarray,
    alpha: float,
    min_support: float,
    max_support: float,
) -> dict | None:
    support = float(mask.mean())
    if support < min_support or support > max_support:
        return None
    if int(mask.sum()) < 3:
        return None
    divergence = _trajectory_divergence(trajectories[mask], trajectories)
    quality = (support**alpha) * divergence
    return {
        "selectors": selectors,
        "mask": mask,
        "support": support,
        "n_entities": int(mask.sum()),
        "divergence": float(divergence),
        "quality": float(quality),
    }


def _trajectory_divergence(subgroup: np.ndarray, population: np.ndarray) -> float:
    values = []
    for j in range(population.shape[1]):
        if wasserstein_distance is not None:
            values.append(float(wasserstein_distance(subgroup[:, j], population[:, j])))
        else:
            values.append(float(abs(np.nanmean(subgroup[:, j]) - np.nanmean(population[:, j]))))
    return float(np.nanmean(values))


def _select_diverse_rules(rules: list[dict], top_k: int, max_jaccard: float) -> list[dict]:
    selected = []
    for rule in rules:
        duplicate = False
        for prior in selected:
            intersection = np.logical_and(rule["mask"], prior["mask"]).sum()
            union = np.logical_or(rule["mask"], prior["mask"]).sum()
            jaccard = intersection / union if union else 0.0
            if jaccard > max_jaccard:
                duplicate = True
                break
        if duplicate:
            continue
        selected.append(rule)
        if len(selected) >= top_k:
            break
    return selected


def _rules_to_frame(rules: list[dict], entity_ids: list[str]) -> pd.DataFrame:
    rows = []
    entity_arr = np.asarray(entity_ids, dtype=object)
    for i, rule in enumerate(rules):
        members = entity_arr[rule["mask"]]
        rows.append(
            {
                "rank": i + 1,
                "rule": " AND ".join(s.text() for s in rule["selectors"]),
                "n_conditions": len(rule["selectors"]),
                "n_entities": rule["n_entities"],
                "effective_n_entities": rule.get("effective_n_entities", rule["n_entities"]),
                "support": rule["support"],
                "divergence": rule["divergence"],
                "baseline_effect_size": rule.get("baseline_effect_size"),
                "anomaly_distribution_divergence": rule.get("anomaly_distribution_divergence"),
                "base_quality": rule.get("base_quality"),
                "trajectory_quality": rule.get("trajectory_quality"),
                "label_quality": rule.get("label_quality"),
                "precision": rule.get("precision"),
                "recall": rule.get("recall"),
                "lift": rule.get("lift"),
                "n_interesting_entities": rule.get("n_interesting_entities"),
                "subgroup_iqr": rule.get("subgroup_iqr"),
                "complement_iqr": rule.get("complement_iqr"),
                "concentration_ratio": rule.get("concentration_ratio"),
                "median_shift": rule.get("median_shift"),
                "distribution_location": rule.get("distribution_location"),
                "subgroup_trend_start": rule.get("subgroup_trend_start"),
                "subgroup_trend_end": rule.get("subgroup_trend_end"),
                "subgroup_trend_change": rule.get("subgroup_trend_change"),
                "baseline_trend_start": rule.get("baseline_trend_start"),
                "baseline_trend_end": rule.get("baseline_trend_end"),
                "baseline_trend_change": rule.get("baseline_trend_change"),
                "trend_change_difference": rule.get("trend_change_difference"),
                "trajectory_peak_gap": rule.get("trajectory_peak_gap"),
                "trajectory_peak_gap_index": rule.get("trajectory_peak_gap_index"),
                "subgroup_trend_direction": rule.get("subgroup_trend_direction"),
                "relative_trend_direction": rule.get("relative_trend_direction"),
                "quality": rule["quality"],
                "complexity": rule.get("complexity", len(rule["selectors"])),
                "pareto_optimal": bool(rule.get("pareto_optimal", True)),
                "covering_fallback": bool(rule.get("covering_fallback", False)),
                "source_tree": rule.get("source_tree"),
                "source_leaf": rule.get("source_leaf"),
                "entities": "; ".join(str(x) for x in members[:50]),
            }
        )
    return pd.DataFrame(rows)


def _save_anomaly_scores(entity_ids: list[str], anomaly_scores: np.ndarray, path: Path) -> None:
    pd.DataFrame(
        {
            "entity": entity_ids,
            "trajectory_anomaly_score": anomaly_scores,
        }
    ).to_csv(path, index=False)


def _save_membership(rules: list[dict], entity_ids: list[str], path: Path) -> None:
    df = pd.DataFrame({"entity": entity_ids})
    for i, rule in enumerate(rules):
        df[f"rule_{i + 1:02d}"] = rule["mask"].astype(int)
    df.to_csv(path, index=False)


def _save_rule_plots(
    rules: list[dict],
    centered: np.ndarray,
    baseline: np.ndarray,
    anomaly_scores: np.ndarray,
    entity_target_means: np.ndarray,
    grid: np.ndarray,
    raw_lookup: dict,
    entity_ids: list[str],
    out_dir: Path,
) -> None:
    out_dir = ensure_dir(out_dir)
    x_label, x_ticks, x_ticklabels = _trajectory_time_axis(grid)
    pop_low = np.quantile(centered, 0.25, axis=0)
    pop_high = np.quantile(centered, 0.75, axis=0)
    subgroup_means = [centered[rule["mask"]].mean(axis=0) for rule in rules]
    y_components = [pop_low, pop_high, baseline, *subgroup_means]
    y_low = float(min(np.nanmin(component) for component in y_components))
    y_high = float(max(np.nanmax(component) for component in y_components))
    y_padding = max(0.05 * (y_high - y_low), 1e-9)
    shared_ylim = (y_low - y_padding, y_high + y_padding)
    for i, rule in enumerate(rules):
        subgroup = centered[rule["mask"]]
        sg_mean = subgroup_means[i]
        fig, ax = plt.subplots(figsize=(12, 8.4))
        ax.fill_between(grid, pop_low, pop_high, color="lightgray", alpha=0.6, label="Population IQR")
        ax.plot(grid, baseline, color="black", linewidth=2.0, label="Global mean-centered baseline")
        ax.plot(grid, sg_mean, color="red", linewidth=2.4, label="Rule subgroup mean-centered trend")
        if x_ticks is not None and x_ticklabels is not None:
            ax.set_xticks(x_ticks, x_ticklabels, rotation=30, ha="right")
        ax.set_xlabel(x_label, fontsize=24)
        ax.set_ylabel("Mean-centered target trajectory", fontsize=24)
        ax.set_ylim(*shared_ylim)
        ax.set_title(f"DOLPHIN rule {i + 1}", fontsize=28)
        ax.tick_params(axis="both", labelsize=22)
        ax.legend(loc="upper left", frameon=True, fontsize=20)
        ax.spines[["top", "right"]].set_visible(False)

        fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.12)
        for suffix in [".png", ".svg", ".pdf"]:
            fig.savefig(out_dir / f"rule_{i + 1:02d}_trajectory{suffix}", dpi=300, bbox_inches="tight")
        plt.close(fig)
        _save_subgroup_kde(
            anomaly_scores,
            rule["mask"],
            out_dir / f"rule_{i + 1:02d}_anomaly_score_kde",
            "Mean-centered trajectory anomaly score",
            f"DOLPHIN rule {i + 1}: trend-shape exceptionalness",
        )
        _save_subgroup_kde(
            entity_target_means,
            rule["mask"],
            out_dir / f"rule_{i + 1:02d}_target_mean_kde",
            "Entity mean target level",
            f"DOLPHIN rule {i + 1}: absolute target-level diagnostic",
        )


def _plot_member_summary(entity_ids: list[str], mask: np.ndarray, max_names: int = 18) -> str:
    members = [str(entity_ids[i]) for i in np.flatnonzero(mask)]
    if not members:
        return ""
    numeric_like = sum(_looks_numeric_identifier(member) for member in members[: min(len(members), 25)])
    if numeric_like / min(len(members), 25) > 0.8:
        return ""
    shown = members[:max_names]
    suffix = f", +{len(members) - max_names} more" if len(members) > max_names else ""
    return "Entities: " + "; ".join(shown) + suffix


def _looks_numeric_identifier(value: str) -> bool:
    cleaned = value.strip()
    if not cleaned:
        return True
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _trajectory_time_axis(grid: np.ndarray) -> tuple[str, np.ndarray | None, list[str] | None]:
    finite = np.asarray(grid, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "Time", None, None

    # CMIE monthly panels use pandas Period[M] ordinals. They are small integers
    # such as 648 for 2024-01, unlike calendar years such as 2024.
    if float(np.nanmax(finite)) < 1200:
        tick_count = min(8, len(grid))
        tick_idx = np.linspace(0, len(grid) - 1, tick_count).round().astype(int)
        ticks = np.asarray(grid)[tick_idx]
        labels = []
        for value in ticks:
            try:
                labels.append(pd.Period(ordinal=int(round(float(value))), freq="M").strftime("%Y-%m"))
            except Exception:
                labels.append(f"{value:.0f}")
        return "Month", ticks, labels

    return "Year", None, None


def _save_trajectory_pca_distribution(
    rules: list[dict],
    entity_ids: list[str],
    centered: np.ndarray,
    output_base: Path,
    rules_path: Path,
) -> None:
    if not rules or centered.shape[0] < 3 or centered.shape[1] < 2:
        rules_path.write_text("No trajectory PCA distribution was available.\n", encoding="utf-8")
        return

    output_base = Path(output_base)
    ensure_dir(output_base.parent)
    standardized, scale = _standardize_trajectory_matrix(centered)
    n_components = min(4, standardized.shape[0] - 1, standardized.shape[1])
    if n_components < 1:
        rules_path.write_text("No trajectory PCA distribution was available.\n", encoding="utf-8")
        return

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(standardized)
    component_rows = []
    global_scale = max(_iqr(scores[:, 0]), 1e-12)
    min_explained_variance = 0.05
    best_component = 0
    best_score = -np.inf
    for component in range(n_components):
        values = scores[:, component]
        scale_component = max(_iqr(values), 1e-12)
        separations = []
        subgroup_means = []
        for rule in rules:
            mask = rule["mask"]
            if int(mask.sum()) < 2 or int((~mask).sum()) < 2:
                continue
            separations.append(_wasserstein(values[mask], values[~mask]) / scale_component)
            subgroup_means.append(float(np.mean(values[mask])))
        component_score = float(np.mean(separations)) if separations else 0.0
        component_rows.append(
            {
                "component": f"PC{component + 1}",
                "explained_variance_ratio": float(pca.explained_variance_ratio_[component]),
                "mean_normalized_wasserstein": component_score,
                "population_iqr": float(scale_component),
                "mean_subgroup_score": float(np.mean(subgroup_means)) if subgroup_means else np.nan,
            }
        )
        eligible_component = float(pca.explained_variance_ratio_[component]) >= min_explained_variance
        if eligible_component and component_score > best_score:
            best_score = component_score
            best_component = component
    if best_score < 0:
        best_component = 0

    component_df = pd.DataFrame(component_rows)
    component_df.to_csv(output_base.with_name(output_base.name + "_components.csv"), index=False)

    values = scores[:, best_component]
    xlabel = (
        f"Mean-centered trajectory shape score "
        f"(PC{best_component + 1}, {pca.explained_variance_ratio_[best_component] * 100:.1f}% variance)"
    )
    _save_trajectory_pca_overlay(values, rules, xlabel, output_base)

    entity_arr = np.asarray(entity_ids, dtype=object)
    lines = [
        f"X-axis: {xlabel}. Higher/lower values mean entities differ along the dominant trajectory-shape "
        "direction that best separates the selected subgroups from the population, after mean-centering each entity."
    ]
    for index, rule in enumerate(rules, start=1):
        mask = rule["mask"]
        subgroup = values[mask]
        complement = values[~mask]
        rule_text = rule.get("rule_text") or " AND ".join(selector.text() for selector in rule.get("selectors", []))
        members = [str(x) for x in entity_arr[mask][:50]]
        lines.append(
            f"Rule {index}: mean PC score={float(np.mean(subgroup)):.4g}; "
            f"population mean={float(np.mean(values)):.4g}; "
            f"complement mean={float(np.mean(complement)):.4g}; "
            f"Wasserstein vs complement={_wasserstein(subgroup, complement):.4g}; "
            f"n={int(mask.sum())}; rule={rule_text}\n"
            + "Entities: "
            + "; ".join(members)
        )
    rules_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def _standardize_trajectory_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    scale = np.std(arr, axis=0)
    positive = scale[scale > 1e-12]
    fallback = float(np.median(positive)) if len(positive) else 1.0
    scale = np.where(scale > 1e-12, scale, fallback)
    return arr / scale, scale


def _save_trajectory_pca_overlay(values: np.ndarray, rules: list[dict], xlabel: str, output_base: Path) -> None:
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(13, 7.5))
    all_values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(all_values) > 1 and all_values.nunique() > 1:
        sns.kdeplot(
            all_values,
            ax=ax,
            color="black",
            linewidth=2.4,
            linestyle="--",
            label=f"Full dataset (n={len(all_values)})",
            bw_adjust=1.05,
        )
    colors = plt.get_cmap("tab10").colors
    for index, rule in enumerate(rules, start=1):
        series = pd.Series(values[rule["mask"]]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) <= 1 or series.nunique() <= 1:
            continue
        color = colors[(index - 1) % len(colors)]
        sns.kdeplot(
            series,
            ax=ax,
            color=color,
            linewidth=2.2,
            fill=True,
            alpha=0.18,
            label=f"Subgroup {index} (n={len(series)})",
            bw_adjust=1.0,
        )
        median = float(series.median())
        ax.axvline(median, color=color, linewidth=1.4, alpha=0.75)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Rule-Defined Subgroups on a Trajectory-Shape Distribution", fontsize=16)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_trajectory_signature_distribution(
    rules: list[dict],
    entity_ids: list[str],
    trajectories: np.ndarray,
    output_base: Path,
    rules_path: Path,
) -> None:
    if not rules or trajectories.shape[0] < 3 or trajectories.shape[1] < 4:
        rules_path.write_text("No trajectory signature distribution was available.\n", encoding="utf-8")
        return

    output_base = Path(output_base)
    ensure_dir(output_base.parent)
    signatures = _trajectory_signature_candidates(trajectories)
    axis_name, values, axis_label, axis_scores = _choose_distribution_axis(signatures, rules)
    axis_scores.to_csv(output_base.with_name(output_base.name + "_axis_selection.csv"), index=False)
    _save_signature_distribution_overlay(values, rules, axis_label, output_base)

    entity_arr = np.asarray(entity_ids, dtype=object)
    lines = [
        f"X-axis: {axis_label}. This is an interpretable trajectory summary selected because it best separates "
        "the rule-defined subgroups from their complements among the available trajectory signatures."
    ]
    for index, rule in enumerate(rules, start=1):
        mask = rule["mask"]
        subgroup = values[mask]
        complement = values[~mask]
        rule_text = rule.get("rule_text") or " AND ".join(selector.text() for selector in rule.get("selectors", []))
        members = [str(x) for x in entity_arr[mask][:50]]
        lines.append(
            f"Rule {index}: subgroup median={float(np.median(subgroup)):.4g}; "
            f"population median={float(np.median(values)):.4g}; "
            f"complement median={float(np.median(complement)):.4g}; "
            f"Wasserstein vs complement={_wasserstein(subgroup, complement):.4g}; "
            f"n={int(mask.sum())}; rule={rule_text}\n"
            + "Entities: "
            + "; ".join(members)
        )
    rules_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def _trajectory_signature_candidates(trajectories: np.ndarray) -> dict[str, tuple[np.ndarray, str]]:
    arr = np.asarray(trajectories, dtype=float)
    n_steps = arr.shape[1]
    early_end = max(1, n_steps // 3)
    late_start = min(n_steps - 2, (2 * n_steps) // 3)
    mid = max(1, n_steps // 2)
    start = arr[:, 0]
    early = arr[:, early_end]
    middle = arr[:, mid]
    late = arr[:, late_start]
    end = arr[:, -1]
    diffs = np.diff(arr, axis=1)
    trough = np.min(arr, axis=1)
    peak = np.max(arr, axis=1)
    return {
        "total_change": (end - start, "Total trajectory change"),
        "early_change": (early - start, "Early-period trajectory change"),
        "late_change": (end - late, "Late-period trajectory change"),
        "acceleration": ((end - late) - (early - start), "Trajectory acceleration (late change minus early change)"),
        "middle_to_end_change": (end - middle, "Middle-to-end trajectory change"),
        "volatility": (np.std(diffs, axis=1), "Trajectory volatility (SD of period-to-period changes)"),
        "mean_abs_change": (np.mean(np.abs(diffs), axis=1), "Mean absolute period-to-period change"),
        "max_drawdown": (peak - trough, "Peak-to-trough trajectory range"),
        "recovery_after_trough": (end - trough, "Recovery after lowest trajectory point"),
        "peak_to_end_drop": (peak - end, "Drop from peak to trajectory end"),
    }


def _save_signature_distribution_overlay(values: np.ndarray, rules: list[dict], xlabel: str, output_base: Path) -> None:
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(13, 7.5))
    all_values = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(all_values) > 1 and all_values.nunique() > 1:
        sns.kdeplot(
            all_values,
            ax=ax,
            color="black",
            linewidth=2.4,
            linestyle="--",
            label=f"Full dataset (n={len(all_values)})",
            bw_adjust=1.05,
        )
    colors = plt.get_cmap("tab10").colors
    for index, rule in enumerate(rules, start=1):
        series = pd.Series(values[rule["mask"]]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) <= 1 or series.nunique() <= 1:
            continue
        color = colors[(index - 1) % len(colors)]
        sns.kdeplot(
            series,
            ax=ax,
            color=color,
            linewidth=2.2,
            fill=True,
            alpha=0.18,
            label=f"Subgroup {index} (n={len(series)})",
            bw_adjust=1.0,
        )
        ax.axvline(float(series.median()), color=color, linewidth=1.4, alpha=0.75)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Rule-Defined Subgroups on an Interpretable Trajectory Summary", fontsize=16)
    ax.legend(fontsize=10, frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _save_fixed_trajectory_metric_distributions(
    rules: list[dict],
    entity_ids: list[str],
    centered: np.ndarray,
    baseline: np.ndarray,
    trajectories: np.ndarray,
    output_dir: Path,
) -> None:
    if not rules or trajectories.shape[0] < 3 or trajectories.shape[1] < 2:
        return
    baseline_deviation = _entity_anomaly_scores(centered, baseline, "euclidean")
    mean_abs_change = np.mean(np.abs(np.diff(trajectories, axis=1)), axis=1)
    _save_fixed_metric_distribution(
        values=baseline_deviation,
        rules=rules,
        entity_ids=entity_ids,
        xlabel="Deviation from global mean-centered trajectory baseline",
        description=(
            "X-axis: Euclidean distance between each entity's mean-centered trajectory and the global "
            "mean-centered trajectory baseline. Larger values indicate more unusual trajectory shape."
        ),
        output_base=output_dir / "trajectory_baseline_deviation_distribution",
        rules_path=output_dir / "trajectory_baseline_deviation_distribution_rules.txt",
    )
    _save_fixed_metric_distribution(
        values=mean_abs_change,
        rules=rules,
        entity_ids=entity_ids,
        xlabel="Mean absolute period-to-period target change",
        description=(
            "X-axis: average absolute movement between adjacent points in the trajectory. Larger values "
            "indicate more volatile or more dynamically changing trajectories over the full observation window."
        ),
        output_base=output_dir / "trajectory_mean_abs_change_distribution",
        rules_path=output_dir / "trajectory_mean_abs_change_distribution_rules.txt",
    )


def _save_fixed_metric_distribution(
    values: np.ndarray,
    rules: list[dict],
    entity_ids: list[str],
    xlabel: str,
    description: str,
    output_base: Path,
    rules_path: Path,
) -> None:
    _save_signature_distribution_overlay(values, rules, xlabel, output_base)
    entity_arr = np.asarray(entity_ids, dtype=object)
    lines = [description]
    population_median = float(np.median(values))
    for index, rule in enumerate(rules, start=1):
        mask = rule["mask"]
        subgroup = values[mask]
        complement = values[~mask]
        rule_text = rule.get("rule_text") or " AND ".join(selector.text() for selector in rule.get("selectors", []))
        members = [str(x) for x in entity_arr[mask][:50]]
        lines.append(
            f"Rule {index}: subgroup median={float(np.median(subgroup)):.4g}; "
            f"population median={population_median:.4g}; "
            f"complement median={float(np.median(complement)):.4g}; "
            f"Wasserstein vs complement={_wasserstein(subgroup, complement):.4g}; "
            f"n={int(mask.sum())}; rule={rule_text}\n"
            + "Entities: "
            + "; ".join(members)
        )
    rules_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def _save_syflow_distribution_overview(
    rules: list[dict],
    entity_ids: list[str],
    anomaly_scores: np.ndarray,
    trajectories: np.ndarray,
    output_base: Path,
    rules_path: Path,
) -> None:
    starts = trajectories[:, 0]
    ends = trajectories[:, -1]
    changes = ends - starts
    pct_changes = np.divide(changes * 100.0, np.abs(starts), out=np.full_like(changes, np.nan), where=np.abs(starts) > 1e-12)
    candidates = {
        "entity_mean_target": (trajectories.mean(axis=1), "Entity mean target value across trajectory"),
        "target_start": (starts, "Entity target value at trajectory start"),
        "target_end": (ends, "Entity target value at trajectory end"),
        "target_change": (changes, "Entity start-to-end target change"),
        "target_pct_change": (pct_changes, "Entity start-to-end target change (%)"),
    }
    axis_name, values, axis_label, axis_scores = _choose_distribution_axis(candidates, rules)
    axis_scores.to_csv(output_base.with_name(output_base.name + "_axis_selection.csv"), index=False)
    finite = values[np.isfinite(values)]
    if len(finite) < 2 or not rules:
        rules_path.write_text("No DOLPHIN subgroup distribution rules were available.\n", encoding="utf-8")
        return

    low, high = np.quantile(finite, [0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low, high = float(np.min(finite)), float(np.max(finite))
    bins = np.linspace(low, high, 45)
    valid_rules = []
    rule_lines = []
    for index, rule in enumerate(rules, start=1):
        subgroup = values[rule["mask"]]
        subgroup = subgroup[np.isfinite(subgroup)]
        if len(subgroup) < 2:
            continue
        rule_text = rule.get("rule_text") or " AND ".join(
            selector.text() for selector in rule.get("selectors", [])
        )
        members = [str(entity_ids[i]) for i in np.flatnonzero(rule["mask"])]
        valid_rules.append((index, subgroup, rule_text, members))
        rule_lines.append(f"Rule {index}: {rule_text}\nEntities ({len(members)}): " + "; ".join(members))

    n_cols = 1 if len(valid_rules) == 1 else 2
    n_rows = ceil(len(valid_rules) / n_cols)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5.5 * n_rows), sharex=True)
    axes = np.asarray([axes] if len(valid_rules) == 1 else axes).reshape(-1)
    for panel, (index, subgroup, rule_text, members) in enumerate(valid_rules):
        ax = axes[panel]
        color = colors[(index - 1) % len(colors)]
        ax.hist(
            np.clip(subgroup, low, high),
            bins=bins,
            density=True,
            color=color,
            alpha=0.55,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.set_title(f"Subgroup {index}  |  n={len(subgroup)}", fontsize=15, color=color)
        ax.set_xlabel(axis_label, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        member_text = ", ".join(members[:12])
        if len(members) > 12:
            member_text += f", +{len(members) - 12} more"
        annotation = textwrap.fill(rule_text, width=72) + "\n" + textwrap.fill(
            "Entities: " + member_text, width=86
        )
        ax.text(0.5, -0.30, annotation, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    for ax in axes[len(valid_rules):]:
        ax.axis("off")
    fig.suptitle(f"Rule-Defined Trajectory Subgroups: {axis_label}", fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.78, wspace=0.2)
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_distribution_overlay(values, valid_rules, axis_label, output_base.with_name(output_base.name + "_overlay"))
    rules_path.write_text(
        f"Selected x-axis: {axis_name} ({axis_label})\n\n" + "\n".join(rule_lines) + "\n",
        encoding="utf-8",
    )


def _choose_distribution_axis(candidates: dict, rules: list[dict]) -> tuple[str, np.ndarray, str, pd.DataFrame]:
    records = []
    for name, (raw_values, label) in candidates.items():
        values = np.asarray(raw_values, dtype=float)
        finite = values[np.isfinite(values)]
        scale = _iqr(finite) if len(finite) else 0.0
        separations = []
        for rule in rules:
            mask = rule["mask"] & np.isfinite(values)
            complement = (~rule["mask"]) & np.isfinite(values)
            if mask.sum() >= 2 and complement.sum() >= 2:
                separations.append(_wasserstein(values[mask], values[complement]) / max(scale, 1e-12))
        records.append(
            {
                "axis": name,
                "label": label,
                "mean_normalized_wasserstein": float(np.mean(separations)) if separations else 0.0,
            }
        )
    scores = pd.DataFrame(records).sort_values("mean_normalized_wasserstein", ascending=False).reset_index(drop=True)
    best = scores.iloc[0]
    values, label = candidates[str(best["axis"])]
    return str(best["axis"]), np.asarray(values, dtype=float), label, scores


def _save_distribution_overlay(values: np.ndarray, valid_rules: list[tuple], xlabel: str, output_base: Path) -> None:
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(13, 7.5))
    colors = plt.get_cmap("tab10").colors
    population = pd.Series(values).replace([np.inf, -np.inf], np.nan).dropna()
    if len(population) > 1 and population.nunique() > 1:
        sns.kdeplot(
            population,
            ax=ax,
            color="#222222",
            linewidth=2.2,
            linestyle="--",
            label=f"Entire dataset (n={len(population)})",
            bw_adjust=1.0,
            zorder=1,
        )
    for panel, (index, subgroup, _, _) in enumerate(valid_rules):
        color = colors[panel % len(colors)]
        series = pd.Series(subgroup).replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) > 1 and series.nunique() > 1:
            sns.kdeplot(series, ax=ax, color=color, linewidth=2.2, label=f"Subgroup {index} (n={len(series)})", bw_adjust=1.0)
        else:
            ax.axvline(float(series.iloc[0]), color=color, linewidth=2.2, label=f"Subgroup {index} (n={len(series)})")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Rule-Defined Subgroups Compared with the Entire Dataset", fontsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_subgroup_evidence_summary(
    rules: list[dict],
    entity_ids: list[str],
    trajectories: np.ndarray,
    output_base: Path,
    key_path: Path,
) -> None:
    if not rules:
        key_path.write_text("No DOLPHIN subgroups were selected.\n", encoding="utf-8")
        return
    changes = trajectories[:, -1] - trajectories[:, 0]
    global_change = float(np.mean(changes))
    change_scale = max(_iqr(changes), 1e-12)
    rows = []
    key_lines = []
    for index, rule in enumerate(rules, start=1):
        mask = rule["mask"]
        members = [str(entity_ids[i]) for i in np.flatnonzero(mask)]
        subgroup_change = float(np.mean(changes[mask]))
        rule_text = rule.get("rule_text") or " AND ".join(
            selector.text() for selector in rule.get("selectors", [])
        )
        rows.append(
            {
                "id": f"S{index}",
                "n": int(mask.sum()),
                "support": float(mask.mean()),
                "effect": float(rule.get("baseline_effect_size", np.nan)),
                "change_z": float((subgroup_change - global_change) / change_scale),
                "subgroup_change": subgroup_change,
                "quality": float(rule.get("quality", np.nan)),
            }
        )
        key_lines.append(
            f"S{index} | n={len(members)} | {rule_text}\n"
            + "Entities: "
            + "; ".join(members)
        )
    frame = pd.DataFrame(rows)
    y = np.arange(len(frame))
    fig_height = max(5.2, 0.68 * len(frame) + 2.8)
    fig, (effect_ax, change_ax) = plt.subplots(
        1,
        2,
        figsize=(12.5, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": [1.05, 1.25], "wspace": 0.16},
    )
    sizes = 80.0 + 900.0 * frame["support"].to_numpy()
    quality = frame["quality"].to_numpy(dtype=float)
    quality_min = float(np.nanmin(quality))
    quality_span = max(float(np.nanmax(quality)) - quality_min, 1e-12)
    quality_color = (quality - quality_min) / quality_span
    effect_ax.scatter(
        frame["effect"],
        y,
        s=sizes,
        c=quality_color,
        cmap="viridis",
        vmin=0,
        vmax=1,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    effect_ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1.1, label="Effect threshold")
    effect_ax.set_xlabel("Trajectory effect size (SD)")
    effect_ax.set_yticks(y, frame["id"])
    effect_ax.set_ylabel("Selected subgroup")
    effect_ax.legend(frameon=False, fontsize=9, loc="lower right")

    change_values = frame["change_z"].to_numpy(dtype=float)
    limit = max(1.0, float(np.nanmax(np.abs(change_values))) * 1.15)
    change_ax.axvline(0, color="#555555", linewidth=1.0)
    change_ax.hlines(y, 0, change_values, color="#999999", linewidth=1.5)
    change_ax.scatter(
        change_values,
        y,
        s=sizes,
        c=change_values,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    for row_y, row in zip(y, frame.itertuples(index=False)):
        offset = 0.04 * limit if row.change_z >= 0 else -0.04 * limit
        change_ax.text(
            row.change_z + offset,
            row_y,
            f"n={row.n}; change={row.subgroup_change:+.3g}",
            va="center",
            ha="left" if row.change_z >= 0 else "right",
            fontsize=8.5,
        )
    change_ax.set_xlim(-limit, limit)
    change_ax.set_xlabel("Target-change difference from population (global IQR units)")
    for ax in (effect_ax, change_ax):
        ax.grid(axis="x", alpha=0.22)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.invert_yaxis()
    fig.suptitle("DOLPHIN Subgroup Evidence Summary", fontsize=15)
    fig.text(
        0.5,
        0.025,
        "Bubble area represents subgroup support; effect size measures trajectory-shape separation; "
        "target change is shown relative to the population.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.18, top=0.88, wspace=0.28)
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    key_path.write_text("\n\n".join(key_lines) + "\n", encoding="utf-8")


def _save_subgroup_kde(
    population: np.ndarray,
    mask: np.ndarray,
    output_base: Path,
    xlabel: str,
    title: str,
) -> None:
    import seaborn as sns

    all_values = pd.Series(population).replace([np.inf, -np.inf], np.nan).dropna()
    subgroup_values = pd.Series(population[mask]).replace([np.inf, -np.inf], np.nan).dropna()
    complement_values = pd.Series(population[~mask]).replace([np.inf, -np.inf], np.nan).dropna()

    fig = plt.figure(figsize=(12, 7))
    if len(all_values) > 1 and all_values.nunique() > 1:
        sns.kdeplot(all_values, color="black", linewidth=2, label="Global population", bw_adjust=1.1)
    if len(complement_values) > 1 and complement_values.nunique() > 1:
        sns.kdeplot(complement_values, color="gray", fill=True, alpha=0.2, label="Entities outside rule", bw_adjust=1.1)
    if len(subgroup_values) > 1 and subgroup_values.nunique() > 1:
        sns.kdeplot(subgroup_values, color="red", fill=True, alpha=0.3, label="Rule subgroup", bw_adjust=1.1)
        sns.rugplot(subgroup_values, color="red", height=0.03, alpha=0.35)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _natural_language_rules(rules_df: pd.DataFrame) -> str:
    if rules_df.empty:
        return "No DOLPHIN rules were selected.\n"
    lines = []
    for row in rules_df.itertuples(index=False):
        lines.append(
            "Rule {rank}: IF {rule}, THEN the matched entities have a mean-centered target trend "
            "that differs from the global mean-centered trajectory baseline. The subgroup contains "
            "{n_entities} entities ({support:.3f} support; effective weighted size {effective_n:.2f}), "
            "with trajectory-shape divergence {divergence:.4g}, complexity {complexity}, "
            "distribution location '{location}', median shift {median_shift:.4g}, "
            "relative concentration {concentration:.3f}, and separation quality {quality:.4g}. "
            "Trajectory reason: the subgroup trend is {direction}; its start-to-end mean-centered "
            "target change is {subgroup_change:+.4g}, compared with baseline change {baseline_change:+.4g} "
            "({relative_direction}; difference {change_difference:+.4g}).".format(
                rank=getattr(row, "rank"),
                rule=getattr(row, "rule"),
                n_entities=getattr(row, "n_entities"),
                effective_n=getattr(row, "effective_n_entities"),
                support=getattr(row, "support"),
                divergence=getattr(row, "divergence"),
                complexity=getattr(row, "complexity"),
                location=getattr(row, "distribution_location"),
                median_shift=getattr(row, "median_shift"),
                concentration=getattr(row, "concentration_ratio"),
                quality=getattr(row, "quality"),
                direction=getattr(row, "subgroup_trend_direction", "unknown"),
                subgroup_change=getattr(row, "subgroup_trend_change", np.nan),
                baseline_change=getattr(row, "baseline_trend_change", np.nan),
                relative_direction=getattr(row, "relative_trend_direction", "unknown relative change"),
                change_difference=getattr(row, "trend_change_difference", np.nan),
            )
        )
    return "\n".join(lines) + "\n"


def _rule_key(selectors: list[Selector]) -> tuple:
    return tuple(sorted((s.feature, s.op, round(float(s.threshold), 10)) for s in selectors))


def _pretty_feature_name(name: str) -> str:
    label = name
    unit = _PRETTY_TIME_UNIT
    if label.startswith("latest_"):
        label = "latest " + label[len("latest_"):]
    else:
        patterns = [
            (r"^lag(\d+)_(.+)$", "{n}-{unit} lag {rest}"),
            (r"^delta(\d+)_(.+)$", "{n}-{unit} change in {rest}"),
            (r"^roll_mean(\d+)_(.+)$", "{n}-{unit} mean {rest}"),
            (r"^roll_std(\d+)_(.+)$", "{n}-{unit} volatility {rest}"),
        ]
        for pattern, template in patterns:
            match = re.match(pattern, label)
            if match:
                n, rest = match.groups()
                label = template.format(n=n, unit=unit, rest=rest)
                break
    replacements = {
        "MAX_EDU_LEVEL": "maximum education level",
        "TOT_N": "household size",
        "EMPLOYED_N": "number of employed members",
        "dominant_source_business_dominant": "business-dominant income source",
        "dominant_source_capital_dominant": "capital-dominant income source",
        "dominant_source_mixed": "mixed income source",
        "dominant_source_transfer_dominant": "transfer-dominant income source",
        "dominant_source_wage_dominant": "wage-dominant income source",
        "income_quintile_Q1": "baseline income quintile Q1",
        "income_quintile_Q2": "baseline income quintile Q2",
        "income_quintile_Q3": "baseline income quintile Q3",
        "income_quintile_Q4": "baseline income quintile Q4",
        "income_quintile_Q5": "baseline income quintile Q5",
    }
    for raw, pretty in replacements.items():
        label = label.replace(raw, pretty)
    return label.replace("_", " ").replace("%", " percent").replace("goverment", "government")
