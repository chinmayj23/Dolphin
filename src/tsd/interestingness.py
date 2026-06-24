from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

LABEL_COL = "is_interesting"


def tail_cluster(table: pd.DataFrame, id_col: str, target_col: str, target_cfg: dict, cfg: dict) -> pd.DataFrame:
    out = table.copy()
    y = pd.to_numeric(out[target_col], errors="coerce")
    mask = np.zeros(len(out), dtype=bool)
    valid = y.notna().to_numpy()
    if valid.sum() >= 3:
        yy = y[valid].to_numpy().reshape(-1, 1)
        try:
            z = linkage(yy, method="single")
            heights = z[:, 2]
            diffs = np.diff(heights)
            threshold = heights[int(np.argmax(diffs))] if len(diffs) else 0.0
            clusters = fcluster(z, t=threshold, criterion="distance")
            labels, counts = np.unique(clusters, return_counts=True)
            normal = labels[int(np.argmax(counts))]
            mask[np.where(valid)[0]] = clusters != normal
        except Exception:
            mask[:] = False

    high_q = float(cfg.get("high_value_quantile", 0.975))
    upper = y.quantile(high_q)
    mask |= (y >= upper).fillna(False).to_numpy()
    if bool(target_cfg.get("two_tailed", False)):
        lower = y.quantile(1.0 - high_q)
        mask |= (y <= lower).fillna(False).to_numpy()
    else:
        lower = np.nan

    out[LABEL_COL] = mask.astype(int)
    out["interesting_score"] = y
    out["state"] = np.where(mask, "tail", "normal")
    out["transition"] = ""
    out["method"] = "tail_cluster"
    out["threshold_high"] = upper
    out["threshold_low"] = lower
    out = add_numeric_transition_fields(out, id_col=id_col, target_col=target_col, target_cfg=target_cfg, cfg=cfg)
    return out


def numeric_transition(table: pd.DataFrame, id_col: str, target_col: str, target_cfg: dict, cfg: dict) -> pd.DataFrame:
    out = table.sort_values([id_col, "_year"]).copy()
    out = add_numeric_transition_fields(out, id_col=id_col, target_col=target_col, target_cfg=target_cfg, cfg=cfg)
    valid = out["next_target"].notna()
    label_score = out["transition_score"]
    out["interesting_score"] = label_score
    score_quantile = float(cfg.get("score_quantile", 0.975))
    threshold = label_score.loc[valid].quantile(score_quantile)
    label = (label_score > threshold) & valid
    if not label.any():
        n_valid = int(valid.sum())
        n_select = max(1, int(np.ceil((1.0 - score_quantile) * n_valid)))
        top_idx = label_score.loc[valid].nlargest(n_select).index
        label = pd.Series(False, index=out.index)
        label.loc[top_idx] = True
    out[LABEL_COL] = label.astype(int)
    out["state"] = np.where(out["transition_z"] > 0, "increase", "decrease")
    out.loc[out["transition_z"].abs() < float(cfg.get("stable_z", 0.5)), "state"] = "stable"
    out["transition"] = out["state"]
    out["method"] = "numeric_transition"
    out["threshold_high"] = threshold
    out["threshold_low"] = np.nan
    return out


def add_numeric_transition_fields(
    table: pd.DataFrame,
    id_col: str,
    target_col: str,
    target_cfg: dict,
    cfg: dict,
) -> pd.DataFrame:
    out = table.copy()
    if not id_col or id_col not in out.columns:
        return out

    out = out.sort_values([id_col, "_year"]).copy()
    y = pd.to_numeric(out[target_col], errors="coerce")
    out["next_target"] = out.groupby(id_col, sort=False)[target_col].shift(-1)
    out["next_year"] = out.groupby(id_col, sort=False)["_year"].shift(-1)
    out["transition_delta"] = pd.to_numeric(out["next_target"], errors="coerce") - y

    delta = pd.to_numeric(out["transition_delta"], errors="coerce")
    center_by_year = delta.groupby(out["_year"]).transform("median")
    mad_by_year = (delta - center_by_year).abs().groupby(out["_year"]).transform("median")
    global_mad = float((delta - delta.median()).abs().median() or delta.std() or 1.0)
    scale = 1.4826 * mad_by_year.replace(0, np.nan).fillna(global_mad)
    out["expected_delta"] = center_by_year
    out["transition_residual"] = delta - center_by_year
    out["transition_z"] = out["transition_residual"] / scale

    direction = str(cfg.get("direction", "auto"))
    if direction == "auto":
        direction = "two_sided" if bool(target_cfg.get("two_tailed", False)) else "positive"
    if direction == "positive":
        out["transition_score"] = out["transition_z"]
    elif direction == "negative":
        out["transition_score"] = -out["transition_z"]
    else:
        out["transition_score"] = out["transition_z"].abs()

    out["transition"] = np.where(out["transition_z"] > 0, "increase", "decrease")
    out.loc[out["transition_z"].abs() < float(cfg.get("stable_z", 0.5)), "transition"] = "stable"
    return out
