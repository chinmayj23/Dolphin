from __future__ import annotations

from pathlib import Path

import pandas as pd

from .interestingness import LABEL_COL


def save_distribution_plot(labeled: pd.DataFrame, target_col: str, output_base: Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_base.parent.mkdir(parents=True, exist_ok=True)
    target_series = pd.to_numeric(labeled[target_col], errors="coerce")
    interesting = labeled[labeled[LABEL_COL].astype(bool)]
    non_interesting = labeled[~labeled[LABEL_COL].astype(bool)]

    fig = plt.figure(figsize=(12, 7))
    sns.kdeplot(
        target_series,
        color="blue",
        fill=True,
        alpha=0.25,
        label="Original",
        bw_adjust=1.1,
        common_norm=False,
    )

    non_interesting_target = pd.to_numeric(non_interesting[target_col], errors="coerce").dropna()
    interesting_target = pd.to_numeric(interesting[target_col], errors="coerce").dropna()

    if len(non_interesting_target) > 1:
        sns.kdeplot(
            non_interesting_target,
            color="green",
            fill=True,
            alpha=0.25,
            label="Non-Interesting",
            bw_adjust=1.1,
            common_norm=False,
        )
        sns.rugplot(non_interesting_target, color="green", height=0.02, alpha=0.15)

    if len(interesting_target) > 1:
        sns.kdeplot(
            interesting_target,
            color="red",
            fill=True,
            alpha=0.25,
            label="Interesting",
            bw_adjust=1.1,
            common_norm=False,
        )
        sns.rugplot(interesting_target, color="red", height=0.03, alpha=0.25)

    plt.xlabel(target_col)
    plt.ylabel("Density")
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.tight_layout()
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_numeric_transition_plots(labeled: pd.DataFrame, output_dir: Path) -> None:
    for col, label in [
        ("transition_delta", "Transition delta"),
        ("transition_z", "Transition z-score"),
        ("transition_score", "Transition score"),
        ("interesting_score", "Method interestingness score"),
    ]:
        if col in labeled.columns:
            _save_labeled_kde(labeled, col, output_dir / f"{col}_kde", label)


def _save_labeled_kde(labeled: pd.DataFrame, value_col: str, output_base: Path, xlabel: str) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_base.parent.mkdir(parents=True, exist_ok=True)
    all_values = pd.to_numeric(labeled[value_col], errors="coerce").dropna()
    interesting_values = pd.to_numeric(
        labeled.loc[labeled[LABEL_COL].astype(bool), value_col],
        errors="coerce",
    ).dropna()
    non_interesting_values = pd.to_numeric(
        labeled.loc[~labeled[LABEL_COL].astype(bool), value_col],
        errors="coerce",
    ).dropna()

    fig = plt.figure(figsize=(12, 7))
    if len(all_values) > 1:
        sns.kdeplot(all_values, color="blue", fill=True, alpha=0.25, label="Original", bw_adjust=1.1, common_norm=False)
    if len(non_interesting_values) > 1:
        sns.kdeplot(non_interesting_values, color="green", fill=True, alpha=0.25, label="Non-Interesting", bw_adjust=1.1, common_norm=False)
        sns.rugplot(non_interesting_values, color="green", height=0.02, alpha=0.12)
    if len(interesting_values) > 1:
        sns.kdeplot(interesting_values, color="red", fill=True, alpha=0.25, label="Interesting", bw_adjust=1.1, common_norm=False)
        sns.rugplot(interesting_values, color="red", height=0.03, alpha=0.25)

    plt.xlabel(xlabel)
    plt.ylabel("Density")
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys())
    plt.tight_layout()
    for suffix in [".png", ".svg", ".pdf"]:
        fig.savefig(output_base.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)
