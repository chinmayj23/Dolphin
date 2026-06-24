from __future__ import annotations

from pathlib import Path
from math import ceil
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import _tree, export_text, plot_tree

try:
    from scipy.stats import wasserstein_distance
except Exception:  # pragma: no cover
    wasserstein_distance = None

from .interestingness import LABEL_COL
from .io import ensure_dir, save_json
from .metrics import classification_metrics
from .tree_viz import save_graphviz_style_tree


def run_rf_module(
    spans: pd.DataFrame,
    id_col: str,
    feature_names: list[str],
    split_cfg: dict,
    rf_cfg: dict,
    output_dir: Path,
    method_name: str = "",
) -> dict:
    output_dir = ensure_dir(output_dir)
    tree_dir = ensure_dir(output_dir / "forest_trees")
    usable = [c for c in feature_names if c in spans.columns]
    X = spans[usable].apply(pd.to_numeric, errors="coerce")
    keep = X.isna().mean(axis=0) < 0.95
    X = X.loc[:, keep]
    usable = X.columns.tolist()
    display_names = [_pretty_feature_name(c) for c in usable]
    y = spans[LABEL_COL].astype(int).to_numpy()
    if len(usable) == 0 or len(np.unique(y)) < 2:
        payload = {"error": "Need at least one feature and both classes."}
        save_json(output_dir / "rule_metrics.json", payload)
        pd.DataFrame().to_csv(output_dir / "rules.csv", index=False)
        (output_dir / "rules_natural_language.txt").write_text("No positive rules were exported.\n", encoding="utf-8")
        return payload

    subjects = spans[id_col].astype(str).to_numpy()
    train_subjects, test_subjects = train_test_split(
        np.unique(subjects),
        test_size=float(split_cfg.get("test_size", 0.25)),
        random_state=int(split_cfg.get("seed", 42)),
    )
    train_mask = np.isin(subjects, train_subjects)
    test_mask = np.isin(subjects, test_subjects)

    forest = RandomForestClassifier(
        n_estimators=int(rf_cfg.get("n_estimators", 80)),
        max_depth=int(rf_cfg.get("max_depth", 4)),
        min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 12)),
        max_features=rf_cfg.get("max_features", "sqrt"),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        random_state=int(rf_cfg.get("seed", 42)),
        n_jobs=1,
    )
    pipe = make_pipeline(SimpleImputer(strategy="median"), forest)
    pipe.fit(X.iloc[train_mask], y[train_mask])
    tree_scores = []
    X_imp_train = pipe.named_steps["simpleimputer"].transform(X.iloc[train_mask])
    X_imp_test = pipe.named_steps["simpleimputer"].transform(X.iloc[test_mask])
    X_imp_all = pipe.named_steps["simpleimputer"].transform(X)
    for i, tree in enumerate(forest.estimators_):
        tree_pred = tree.predict(X_imp_test)
        m = classification_metrics(y[test_mask], tree_pred)
        tree_scores.append({"tree_index": i, **m, "n_nodes": int(tree.tree_.node_count), "depth": int(tree.tree_.max_depth)})
    tree_summary = pd.DataFrame(tree_scores).sort_values(["balanced_accuracy", "f1"], ascending=False)
    tree_summary.to_csv(output_dir / "forest_summary.csv", index=False)

    selected_indices, selected_metrics, selection_df = _select_compact_forest(
        forest,
        tree_summary,
        X_imp_test,
        y[test_mask],
        tolerance=float(rf_cfg.get("selection_accuracy_tolerance", 0.02)),
        min_trees=int(rf_cfg.get("min_selected_trees", 1)),
        max_trees=int(rf_cfg.get("max_selected_trees", len(forest.estimators_))),
    )
    selection_df.to_csv(output_dir / "forest_selection.csv", index=False)
    metrics = {
        **selected_metrics,
        "n_train_spans": int(train_mask.sum()),
        "n_test_spans": int(test_mask.sum()),
        "n_features": int(len(usable)),
        "n_candidate_trees": int(len(forest.estimators_)),
        "n_selected_trees": int(len(selected_indices)),
        "selection_accuracy_tolerance": float(rf_cfg.get("selection_accuracy_tolerance", 0.02)),
        "selected_tree_indices": [int(i) for i in selected_indices],
    }

    all_rules = []
    overview_rules = []
    forest_leaf_summaries = {}
    for rank, tree_index in enumerate(selected_indices):
        tree = forest.estimators_[tree_index]
        prefix = tree_dir / f"tree_{rank:02d}_idx_{tree_index:03d}"
        text = export_text(tree, feature_names=display_names, decimals=3, max_depth=int(rf_cfg.get("max_depth", 4)))
        leaf_summary = _leaf_transition_summary(tree, X_imp_train, spans.iloc[train_mask].reset_index(drop=True))
        forest_leaf_summaries[int(tree_index)] = leaf_summary
        text = _append_leaf_summary(text, leaf_summary)
        (prefix.with_suffix(".txt")).write_text(text, encoding="utf-8")
        _save_tree_figure(tree, display_names, prefix, leaf_summary=leaf_summary)
        rules = _extract_positive_rules(tree, display_names, leaf_summary=leaf_summary)
        rules["tree_rank"] = rank
        rules["tree_index"] = tree_index
        all_rules.append(rules)
        leaf_ids_all = tree.apply(X_imp_all)
        for row in rules.itertuples(index=False):
            overview_rules.append(
                {
                    "rule": str(row.rule),
                    "mask": leaf_ids_all == int(row.leaf_id),
                }
            )

    pd.DataFrame(
        [
            {
                "tree_rank": rank + 1,
                "forest_index": int(tree_index),
                "depth": int(forest.estimators_[tree_index].tree_.max_depth),
                "leaves": int(forest.estimators_[tree_index].tree_.n_leaves),
                "png": f"tree_{rank:02d}_idx_{tree_index:03d}.png",
                "pdf": f"tree_{rank:02d}_idx_{tree_index:03d}.pdf",
                "svg": f"tree_{rank:02d}_idx_{tree_index:03d}.svg",
                "text": f"tree_{rank:02d}_idx_{tree_index:03d}.txt",
            }
            for rank, tree_index in enumerate(selected_indices)
        ]
    ).to_csv(tree_dir / "manifest.csv", index=False)

    rules_df = pd.concat(all_rules, ignore_index=True) if all_rules else pd.DataFrame()
    if method_name == "numeric_transition":
        rules_df, overview_rules = _select_distribution_rules(
            forest,
            selected_indices,
            X_imp_all,
            spans.reset_index(drop=True),
            display_names,
            rf_cfg,
        )
    cols = [
        "tree_rank",
        "tree_index",
        "leaf_id",
        "rule",
        "positive_weight",
        "total_weight",
        "positive_rate",
        "transition_direction",
        "mean_transition_delta",
        "mean_target_pct_change",
        "mean_transition_z",
        "mean_score",
        "mean_span_length",
        "mean_start_year",
        "mean_end_year",
        "n_training_spans",
        "n_conditions",
        "support",
        "distribution_divergence",
        "subgroup_iqr",
        "complement_iqr",
        "concentration_ratio",
        "median_shift",
        "distribution_location",
        "distribution_quality",
    ]
    rules_df = rules_df[[c for c in cols if c in rules_df.columns]]
    rules_df.to_csv(output_dir / "rules.csv", index=False)
    (output_dir / "rules_natural_language.txt").write_text(
        _natural_language_rules(rules_df, distributional=method_name == "numeric_transition"),
        encoding="utf-8",
    )
    _save_rule_kde_plots(forest, selected_indices, X_imp_all, spans.reset_index(drop=True), output_dir / "rule_plots")
    _save_subgroup_distribution_overview(
        spans.reset_index(drop=True),
        overview_rules,
        output_dir / "subgroup_distribution",
        output_dir / "subgroup_distribution_rules.txt",
    )
    save_json(output_dir / "rule_metrics.json", metrics)
    return metrics


def _select_distribution_rules(
    forest,
    selected_indices: list[int],
    X_all: np.ndarray,
    spans: pd.DataFrame,
    feature_names: list[str],
    cfg: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    score_col = "transition_z" if "transition_z" in spans else "transition_delta"
    values = pd.to_numeric(spans[score_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    candidates = []
    seen = set()
    min_support = float(cfg.get("distribution_rule_min_support", 0.03))
    max_support = float(cfg.get("distribution_rule_max_support", 0.4))
    support_exp = float(cfg.get("distribution_support_exponent", 0.25))
    concentration_exp = float(cfg.get("distribution_concentration_exponent", 0.5))
    max_concentration = float(cfg.get("distribution_max_concentration_ratio", 4.0))

    for tree_rank, tree_index in enumerate(selected_indices):
        tree = forest.estimators_[tree_index]
        leaf_ids = tree.apply(X_all)
        conditions_by_leaf = _all_leaf_conditions(tree, feature_names)
        for leaf_id, conditions in conditions_by_leaf.items():
            mask = (leaf_ids == leaf_id) & valid
            complement = (~mask) & valid
            support = float(mask.sum() / valid.sum()) if valid.sum() else 0.0
            if support < min_support or support > max_support or mask.sum() < 3 or complement.sum() < 3:
                continue
            key = tuple(mask.tolist())
            if key in seen:
                continue
            seen.add(key)
            subgroup = values[mask]
            outside = values[complement]
            divergence = _distribution_wasserstein(subgroup, outside)
            subgroup_iqr = _distribution_iqr(subgroup)
            complement_iqr = _distribution_iqr(outside)
            global_iqr = max(_distribution_iqr(values[valid]), 1e-12)
            concentration = min(
                max_concentration,
                (complement_iqr + 0.05 * global_iqr) / (subgroup_iqr + 0.05 * global_iqr),
            )
            median_shift = float(np.median(subgroup) - np.median(outside))
            location = _distribution_location(median_shift, global_iqr)
            quality = (support**support_exp) * divergence * (concentration**concentration_exp)
            summary = _span_mask_summary(spans, mask)
            candidates.append(
                {
                    "tree_rank": tree_rank,
                    "tree_index": tree_index,
                    "leaf_id": int(leaf_id),
                    "rule": " AND ".join(conditions) if conditions else "TRUE",
                    "n_conditions": len(conditions),
                    "support": support,
                    "distribution_divergence": divergence,
                    "subgroup_iqr": subgroup_iqr,
                    "complement_iqr": complement_iqr,
                    "concentration_ratio": concentration,
                    "median_shift": median_shift,
                    "distribution_location": location,
                    "distribution_quality": quality,
                    "mask": mask,
                    **summary,
                }
            )

    pareto = []
    for rule in candidates:
        dominated = any(
            other["n_conditions"] < rule["n_conditions"]
            and other["distribution_quality"] >= rule["distribution_quality"]
            for other in candidates
        )
        if not dominated:
            pareto.append(rule)
    ranked = sorted(pareto or candidates, key=lambda r: r["distribution_quality"], reverse=True)
    selected = []
    max_jaccard = float(cfg.get("distribution_max_jaccard", 0.9))
    for rule in ranked:
        if any(_distribution_jaccard(rule["mask"], prior["mask"]) > max_jaccard for prior in selected):
            continue
        selected.append(rule)
        if len(selected) >= int(cfg.get("distribution_rule_top_k", 10)):
            break
    overview = [{"rule": r["rule"], "mask": r["mask"]} for r in selected]
    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "mask"} for r in selected])
    return frame, overview


def _all_leaf_conditions(tree, feature_names: list[str]) -> dict[int, list[str]]:
    result = {}
    t = tree.tree_

    def walk(node: int, conditions: list[str]) -> None:
        if t.feature[node] == _tree.TREE_UNDEFINED:
            result[int(node)] = conditions
            return
        feature = feature_names[int(t.feature[node])]
        threshold = float(t.threshold[node])
        walk(int(t.children_left[node]), conditions + [f"{feature} <= {threshold:.4g}"])
        walk(int(t.children_right[node]), conditions + [f"{feature} > {threshold:.4g}"])

    walk(0, [])
    return result


def _span_mask_summary(spans: pd.DataFrame, mask: np.ndarray) -> dict:
    group = spans.loc[mask]
    def mean(col: str) -> float:
        return float(pd.to_numeric(group[col], errors="coerce").mean()) if col in group else np.nan
    delta = mean("transition_delta")
    return {
        "transition_direction": "high/increasing transition" if delta > 0 else "low/decreasing transition" if delta < 0 else "flat transition",
        "mean_transition_delta": delta,
        "mean_target_pct_change": mean("target_pct_change"),
        "mean_transition_z": mean("transition_z"),
        "mean_score": mean("score_mean"),
        "mean_span_length": mean("span_length"),
        "mean_start_year": mean("start_year"),
        "mean_end_year": mean("end_year"),
        "n_training_spans": int(mask.sum()),
    }


def _distribution_wasserstein(left: np.ndarray, right: np.ndarray) -> float:
    if wasserstein_distance is not None:
        return float(wasserstein_distance(left, right))
    return float(abs(np.mean(left) - np.mean(right)))


def _distribution_iqr(values: np.ndarray) -> float:
    return float(np.quantile(values, 0.75) - np.quantile(values, 0.25))


def _distribution_location(median_shift: float, scale: float) -> str:
    if abs(median_shift) <= 0.1 * scale:
        return "shape/dispersion shifted"
    return "upper shifted" if median_shift > 0 else "lower shifted"


def _distribution_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def _save_subgroup_distribution_overview(
    spans: pd.DataFrame,
    rules: list[dict],
    output_base: Path,
    rules_path: Path,
) -> None:
    candidate_labels = {
        "target_mean": "Mean target value within entity time span",
        "target_start": "Target value at span start",
        "target_end": "Target value at span end",
        "transition_delta": "Start-to-end target change",
        "target_pct_change": "Start-to-end target change (%)",
    }
    axis_records = []
    for column, label in candidate_labels.items():
        if column not in spans:
            continue
        candidate = pd.to_numeric(spans[column], errors="coerce").to_numpy(dtype=float)
        finite_candidate = candidate[np.isfinite(candidate)]
        scale = float(np.quantile(finite_candidate, 0.75) - np.quantile(finite_candidate, 0.25)) if len(finite_candidate) else 0.0
        separations = []
        for rule in rules:
            mask = np.asarray(rule["mask"], dtype=bool) & np.isfinite(candidate)
            complement = (~np.asarray(rule["mask"], dtype=bool)) & np.isfinite(candidate)
            if mask.sum() >= 2 and complement.sum() >= 2:
                separations.append(_distribution_wasserstein(candidate[mask], candidate[complement]) / max(scale, 1e-12))
        axis_records.append(
            {
                "axis": column,
                "label": label,
                "mean_normalized_wasserstein": float(np.mean(separations)) if separations else 0.0,
            }
        )
    axis_scores = pd.DataFrame(axis_records).sort_values("mean_normalized_wasserstein", ascending=False).reset_index(drop=True)
    if axis_scores.empty:
        rules_path.write_text("No target-derived distribution axis was available.\n", encoding="utf-8")
        return
    selected_axis = str(axis_scores.iloc[0]["axis"])
    xlabel = str(axis_scores.iloc[0]["label"])
    axis_scores.to_csv(output_base.with_name(output_base.name + "_axis_selection.csv"), index=False)
    values = pd.to_numeric(spans[selected_axis], errors="coerce")
    finite = values[np.isfinite(values)]
    if len(finite) < 2 or not rules:
        rules_path.write_text("No subgroup distribution rules were available.\n", encoding="utf-8")
        return

    low, high = np.quantile(finite, [0.005, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low, high = float(finite.min()), float(finite.max())
    bins = np.linspace(low, high, 45)
    valid_rules = []
    rule_lines = []
    for index, rule in enumerate(rules, start=1):
        subgroup = values[rule["mask"]]
        subgroup = subgroup[np.isfinite(subgroup)]
        if len(subgroup) < 2:
            continue
        valid_rules.append((index, rule, subgroup))
        rule_lines.append(f"Rule {index}: {rule['rule']}")

    n_cols = 1 if len(valid_rules) == 1 else 2
    n_rows = ceil(len(valid_rules) / n_cols)
    colors = plt.get_cmap("tab10").colors
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9 * n_cols, 5.5 * n_rows), sharex=True)
    axes = np.asarray([axes] if len(valid_rules) == 1 else axes).reshape(-1)
    for panel, (index, rule, subgroup) in enumerate(valid_rules):
        ax = axes[panel]
        color = colors[(index - 1) % len(colors)]
        ax.hist(
            subgroup.clip(low, high),
            bins=bins,
            density=True,
            color=color,
            alpha=0.55,
            edgecolor=color,
            linewidth=1.5,
        )
        ax.set_title(f"Subgroup {index}  |  n={len(subgroup)}", fontsize=15, color=color)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        wrapped = textwrap.fill(str(rule["rule"]), width=72)
        ax.text(0.5, -0.27, wrapped, transform=ax.transAxes, ha="center", va="top", fontsize=10)
    for ax in axes[len(valid_rules):]:
        ax.axis("off")
    fig.suptitle("Rule-Defined Subgroups", fontsize=18)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, hspace=0.62, wspace=0.2)
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
    _save_rf_distribution_overlay(valid_rules, xlabel, output_base.with_name(output_base.name + "_overlay"))
    rules_path.write_text(
        f"Selected x-axis: {selected_axis} ({xlabel})\n\n" + "\n".join(rule_lines) + "\n",
        encoding="utf-8",
    )


def _save_rf_distribution_overlay(valid_rules: list[tuple], xlabel: str, output_base: Path) -> None:
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(13, 7.5))
    colors = plt.get_cmap("tab10").colors
    for panel, (index, _, subgroup) in enumerate(valid_rules):
        color = colors[panel % len(colors)]
        series = pd.Series(subgroup).replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) > 1 and series.nunique() > 1:
            sns.kdeplot(series, ax=ax, color=color, linewidth=2.2, label=f"Subgroup {index} (n={len(series)})", bw_adjust=1.0)
        elif len(series):
            ax.axvline(float(series.iloc[0]), color=color, linewidth=2.2, label=f"Subgroup {index} (n={len(series)})")
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title("Overlay of All Rule-Defined Subgroup Distributions", fontsize=16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=10, frameon=False)
    fig.tight_layout()
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_rule_kde_plots(forest, selected_indices: list[int], X_all: np.ndarray, spans: pd.DataFrame, out_dir: Path) -> None:
    import seaborn as sns

    out_dir = ensure_dir(out_dir)
    for tree_rank, tree_index in enumerate(selected_indices):
        tree = forest.estimators_[tree_index]
        leaf_ids = tree.apply(X_all)
        for leaf_id in np.unique(leaf_ids):
            weights = tree.tree_.value[int(leaf_id)][0]
            if int(np.argmax(weights)) != 1:
                continue
            mask = leaf_ids == leaf_id
            for value_col, xlabel in [
                ("score_mean", "Transition interestingness score"),
                ("transition_delta", "Mean transition delta"),
                ("target_mean", "Mean target level within span"),
            ]:
                if value_col not in spans:
                    continue
                population = pd.to_numeric(spans[value_col], errors="coerce")
                subgroup = population[mask].dropna()
                complement = population[~mask].dropna()
                population = population.dropna()
                if len(subgroup) < 2 or subgroup.nunique() < 2:
                    continue
                fig = plt.figure(figsize=(12, 7))
                if len(population) > 1 and population.nunique() > 1:
                    sns.kdeplot(population, color="black", linewidth=2, label="Global population", bw_adjust=1.1)
                if len(complement) > 1 and complement.nunique() > 1:
                    sns.kdeplot(complement, color="gray", fill=True, alpha=0.2, label="Entities outside rule", bw_adjust=1.1)
                sns.kdeplot(subgroup, color="red", fill=True, alpha=0.3, label="Rule subgroup", bw_adjust=1.1)
                sns.rugplot(subgroup, color="red", height=0.03, alpha=0.35)
                plt.xlabel(xlabel)
                plt.ylabel("Density")
                plt.title(f"Tree {tree_rank + 1}, leaf {int(leaf_id)}: {xlabel}")
                plt.legend()
                plt.tight_layout()
                base = out_dir / f"tree_{tree_rank:02d}_idx_{tree_index:03d}_leaf_{int(leaf_id):03d}_{value_col}_kde"
                for suffix in [".png", ".svg", ".pdf"]:
                    fig.savefig(base.with_suffix(suffix), dpi=300, bbox_inches="tight")
                plt.close(fig)


def _select_compact_forest(
    forest,
    tree_summary: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    tolerance: float,
    min_trees: int,
    max_trees: int,
) -> tuple[list[int], dict, pd.DataFrame]:
    ranked_indices = [int(x) for x in tree_summary["tree_index"].tolist()]
    max_trees = min(max_trees, len(ranked_indices))
    min_trees = max(1, min(min_trees, max_trees))
    records = []
    best_score = -np.inf
    for k in range(min_trees, max_trees + 1):
        indices = ranked_indices[:k]
        votes = np.vstack([forest.estimators_[i].predict(X_test).astype(int) for i in indices])
        pred = (votes.mean(axis=0) >= 0.5).astype(int)
        m = classification_metrics(y_test, pred)
        records.append({"n_trees": k, "tree_indices": indices, **m})
        best_score = max(best_score, m["balanced_accuracy"])

    selection_df = pd.DataFrame(records)
    eligible = selection_df[selection_df["balanced_accuracy"] >= best_score - tolerance]
    selected = eligible.sort_values(["n_trees", "balanced_accuracy"], ascending=[True, False]).iloc[0]
    selected_indices = [int(x) for x in selected["tree_indices"]]
    selected_metrics = {
        "accuracy": float(selected["accuracy"]),
        "balanced_accuracy": float(selected["balanced_accuracy"]),
        "precision": float(selected["precision"]),
        "recall": float(selected["recall"]),
        "f1": float(selected["f1"]),
        "best_candidate_balanced_accuracy": float(best_score),
    }
    return selected_indices, selected_metrics, selection_df


def _save_tree_figure(tree, feature_names: list[str], base: Path, leaf_summary: dict[int, dict] | None = None) -> None:
    summary = leaf_summary or {}
    save_graphviz_style_tree(
        tree,
        feature_names,
        leaf_label=lambda node: _classification_leaf_label(tree, node, summary.get(node, {})),
        leaf_color=lambda node: "#f3a6a0" if int(np.argmax(tree.tree_.value[node][0])) == 1 else "#b9d8ee",
        title="Interpretable subgroup tree with leaf transition explanations",
        output_base=base,
        fontsize=10,
    )


def _classification_leaf_label(tree, node: int, row: dict) -> str:
    weights = tree.tree_.value[node][0]
    pred = int(np.argmax(weights))
    total = float(weights.sum())
    positive_rate = float(weights[1] / total) if total else 0.0
    return (
        ("INTERESTING" if pred == 1 else "NOT INTERESTING")
        + f"\ninteresting rate={100 * positive_rate:.1f}%"
        + f"\n{row.get('transition_direction', 'unknown')}"
        + f"\nmean delta={row.get('mean_transition_delta', np.nan):.3g}"
        + f"\nmean change={row.get('mean_target_pct_change', np.nan):.3g}%"
        + f"\naverage frame={row.get('mean_span_length', np.nan):.2g} years"
    )


def _extract_positive_rules(tree, feature_names: list[str], leaf_summary: dict[int, dict] | None = None) -> pd.DataFrame:
    rows = []
    t = tree.tree_

    def walk(node: int, conditions: list[str]) -> None:
        if t.feature[node] == _tree.TREE_UNDEFINED:
            weights = t.value[node][0]
            pred = int(np.argmax(weights))
            if pred == 1:
                total = float(weights.sum())
                rows.append(
                    {
                        "leaf_id": int(node),
                        "rule": " AND ".join(conditions) if conditions else "TRUE",
                        "positive_weight": float(weights[1]),
                        "total_weight": total,
                        "positive_rate": float(weights[1] / total) if total else 0.0,
                        "n_conditions": int(len(conditions)),
                        **(leaf_summary or {}).get(int(node), {}),
                    }
                )
            return
        feature = feature_names[t.feature[node]]
        threshold = t.threshold[node]
        walk(t.children_left[node], conditions + [f"{feature} <= {threshold:.4g}"])
        walk(t.children_right[node], conditions + [f"{feature} > {threshold:.4g}"])

    walk(0, [])
    return pd.DataFrame(rows)


def _leaf_transition_summary(tree, X_imp_train: np.ndarray, train_spans: pd.DataFrame) -> dict[int, dict]:
    leaf_ids = tree.apply(X_imp_train)
    result = {}
    for leaf_id in np.unique(leaf_ids):
        mask = leaf_ids == leaf_id
        group = train_spans.loc[mask]
        row = {"n_training_spans": int(len(group))}
        if "score_mean" in group:
            row["mean_score"] = float(pd.to_numeric(group["score_mean"], errors="coerce").mean())
        if "transition_delta" in group:
            row["mean_transition_delta"] = float(pd.to_numeric(group["transition_delta"], errors="coerce").mean())
        elif "target_start" in group and "target_end" in group:
            row["mean_transition_delta"] = float(
                (pd.to_numeric(group["target_end"], errors="coerce") - pd.to_numeric(group["target_start"], errors="coerce")).mean()
            )
        if "transition_z" in group:
            row["mean_transition_z"] = float(pd.to_numeric(group["transition_z"], errors="coerce").mean())
        if "target_pct_change" in group:
            row["mean_target_pct_change"] = float(pd.to_numeric(group["target_pct_change"], errors="coerce").mean())
        if "span_length" in group:
            row["mean_span_length"] = float(pd.to_numeric(group["span_length"], errors="coerce").mean())
        if "start_year" in group:
            row["mean_start_year"] = float(pd.to_numeric(group["start_year"], errors="coerce").mean())
        if "end_year" in group:
            row["mean_end_year"] = float(pd.to_numeric(group["end_year"], errors="coerce").mean())
        delta = row.get("mean_transition_delta")
        if delta is None or not np.isfinite(delta):
            direction = "unknown"
        elif delta > 0:
            direction = "high/increasing transition"
        elif delta < 0:
            direction = "low/decreasing transition"
        else:
            direction = "flat transition"
        row["transition_direction"] = direction
        result[int(leaf_id)] = row
    return result


def _append_leaf_summary(text: str, leaf_summary: dict[int, dict]) -> str:
    lines = [text.rstrip(), "", "Leaf transition summaries:"]
    for leaf_id in sorted(leaf_summary):
        row = leaf_summary[leaf_id]
        parts = [
            f"leaf {leaf_id}",
            row.get("transition_direction", "unknown"),
            f"mean_delta={row.get('mean_transition_delta', np.nan):.4g}",
            f"mean_pct={row.get('mean_target_pct_change', np.nan):.3g}%",
            f"avg_frame={row.get('mean_span_length', np.nan):.2g}y",
            f"mean_z={row.get('mean_transition_z', np.nan):.4g}",
            f"n={row.get('n_training_spans', 0)}",
        ]
        lines.append(" | ".join(parts))
    return "\n".join(lines) + "\n"


def _rewrite_tree_labels(artists, tree, feature_names: list[str], leaf_summary: dict[int, dict]) -> None:
    tree_ = tree.tree_
    for artist in artists:
        text = artist.get_text()
        if text in {"True", "False"}:
            artist.set_text("")
            continue
        node_id = None
        for token in text.replace("\n", " ").split():
            if token.startswith("#"):
                try:
                    node_id = int(token[1:])
                    break
                except ValueError:
                    pass
        if node_id is None or node_id >= tree_.node_count:
            continue
        if tree_.feature[node_id] == _tree.TREE_UNDEFINED:
            weights = tree_.value[node_id][0]
            pred = int(np.argmax(weights))
            label = "interesting" if pred == 1 else "not interesting"
            new_text = f"Leaf\n{label}"
            row = leaf_summary.get(node_id, {})
            direction = row.get("transition_direction", "unknown")
            delta = row.get("mean_transition_delta", np.nan)
            pct = row.get("mean_target_pct_change", np.nan)
            span_len = row.get("mean_span_length", np.nan)
            total = float(weights.sum())
            positive_rate = float(weights[1] / total) if total else 0.0
            artist.set_text(
                new_text
                + f"\ninteresting rate={100 * positive_rate:.1f}%"
                + "\n"
                + direction
                + f"\nmean_delta={delta:.3g}"
                + f"\nmean_pct={pct:.3g}%"
                + f"\navg_frame={span_len:.2g}y"
            )
        else:
            feature = feature_names[int(tree_.feature[node_id])]
            threshold = float(tree_.threshold[node_id])
            artist.set_text(f"{feature} <= {threshold:.4g}")


def _pretty_feature_name(name: str) -> str:
    prefixes = [
        ("latest_", "latest "),
        ("lag1_", "1-year lag "),
        ("lag3_", "3-year lag "),
        ("lag5_", "5-year lag "),
        ("delta1_", "1-year change in "),
        ("delta3_", "3-year change in "),
        ("delta5_", "5-year change in "),
        ("roll_mean3_", "3-year mean "),
        ("roll_mean5_", "5-year mean "),
        ("roll_std3_", "3-year volatility "),
        ("roll_std5_", "5-year volatility "),
    ]
    label = name
    for prefix, replacement in prefixes:
        if label.startswith(prefix):
            label = replacement + label[len(prefix):]
            break
    label = label.replace("_", " ")
    label = label.replace("%", " percent")
    label = label.replace("goverment", "government")
    return label


def _natural_language_rules(rules_df: pd.DataFrame, distributional: bool = False) -> str:
    if rules_df.empty:
        return "No rules were exported.\n"
    lines = []
    for row in rules_df.itertuples(index=False):
        if distributional:
            lines.append(
                "Rule {rank}: IF {rule}, THEN the signed transition-score distribution is {location} "
                "relative to entities outside the rule. Support is {support:.3f}, Wasserstein separation "
                "is {divergence:.4g}, median shift is {shift:.4g}, relative concentration is "
                "{concentration:.3f}, and quality is {quality:.4g}.".format(
                    rank=len(lines) + 1,
                    rule=getattr(row, "rule", "TRUE"),
                    location=getattr(row, "distribution_location", "shifted"),
                    support=getattr(row, "support", np.nan),
                    divergence=getattr(row, "distribution_divergence", np.nan),
                    shift=getattr(row, "median_shift", np.nan),
                    concentration=getattr(row, "concentration_ratio", np.nan),
                    quality=getattr(row, "distribution_quality", np.nan),
                )
            )
            continue
        lines.append(
            "Tree {tree_rank}, leaf {leaf_id}: IF {rule}, THEN the span is predicted interesting. "
            "Captured spans show {direction}; mean target change is {delta:.4g} ({pct:.3g}%), "
            "mean transition z-score is {z:.4g}, average frame is {span:.2g} years, "
            "with average years {start:.0f}-{end:.0f}. Leaf positive rate is {precision:.3f}.".format(
                tree_rank=getattr(row, "tree_rank", "?"),
                leaf_id=getattr(row, "leaf_id", "?"),
                rule=getattr(row, "rule", "TRUE"),
                direction=getattr(row, "transition_direction", "unknown"),
                delta=getattr(row, "mean_transition_delta", np.nan),
                pct=getattr(row, "mean_target_pct_change", np.nan),
                z=getattr(row, "mean_transition_z", np.nan),
                span=getattr(row, "mean_span_length", np.nan),
                start=getattr(row, "mean_start_year", np.nan),
                end=getattr(row, "mean_end_year", np.nan),
                precision=getattr(row, "positive_rate", np.nan),
            )
        )
    return "\n".join(lines) + "\n"
