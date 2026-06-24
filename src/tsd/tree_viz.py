from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from sklearn.tree import _tree


def save_graphviz_style_tree(
    estimator,
    feature_names: list[str],
    leaf_label: Callable[[int], str],
    leaf_color: Callable[[int], str],
    title: str,
    output_base: Path,
    fontsize: int = 10,
) -> None:
    tree = estimator.tree_
    positions: dict[int, tuple[float, int]] = {}
    next_leaf = [0]

    def place(node: int, depth: int) -> float:
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == right:
            x = float(next_leaf[0])
            next_leaf[0] += 1
        else:
            x = (place(left, depth + 1) + place(right, depth + 1)) / 2.0
        positions[node] = (x, depth)
        return x

    place(0, 0)
    n_leaves = max(next_leaf[0], 1)
    max_depth = max(depth for _, depth in positions.values())
    x_spacing = 7.2
    y_spacing = 2.9
    fig_width = max(18.0, min(160.0, x_spacing * max(n_leaves, 2)))
    fig_height = max(10.0, y_spacing * (max_depth + 1) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for node, (x, depth) in positions.items():
        x_plot = x * x_spacing
        y_plot = -depth * y_spacing
        left = int(tree.children_left[node])
        right = int(tree.children_right[node])
        if left == right:
            continue
        for child, edge_label in ((left, "True"), (right, "False")):
            child_x, child_depth = positions[child]
            child_x_plot = child_x * x_spacing
            child_y_plot = -child_depth * y_spacing
            ax.plot([x_plot, child_x_plot], [y_plot, child_y_plot], color="#4a4a4a", linewidth=1.4, zorder=1)
            ax.text(
                0.46 * x_plot + 0.54 * child_x_plot,
                0.46 * y_plot + 0.54 * child_y_plot,
                edge_label,
                fontsize=max(fontsize - 2, 8),
                color="#333333",
                ha="center",
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8},
                zorder=2,
            )

    for node, (x, depth) in positions.items():
        x_plot = x * x_spacing
        y_plot = -depth * y_spacing
        is_leaf = tree.feature[node] == _tree.TREE_UNDEFINED
        if is_leaf:
            label = leaf_label(node)
            color = leaf_color(node)
        else:
            feature = feature_names[int(tree.feature[node])]
            label = f"{feature} <= {float(tree.threshold[node]):.4g}"
            color = "white"
        wrapped = "\n".join(textwrap.wrap(label, width=34, break_long_words=False, break_on_hyphens=False))
        ax.text(
            x_plot,
            y_plot,
            wrapped,
            ha="center",
            va="center",
            fontsize=fontsize,
            linespacing=1.22,
            bbox={
                "boxstyle": "round,pad=0.55,rounding_size=0.14",
                "facecolor": color,
                "edgecolor": "#333333",
                "linewidth": 1.25,
            },
            zorder=3,
        )

    ax.set_xlim(-1.2 * x_spacing, (n_leaves - 0.2) * x_spacing)
    ax.set_ylim(-(max_depth + 0.8) * y_spacing, 0.9 * y_spacing)
    ax.axis("off")
    ax.set_title(title, fontsize=max(fontsize + 6, 16), pad=24)
    fig.tight_layout(pad=2.0)
    fig.savefig(output_base.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
