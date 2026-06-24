from __future__ import annotations

import pandas as pd

from .interestingness import LABEL_COL


def build_value_spans(
    labeled: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    feature_names: list[str],
) -> pd.DataFrame:
    rows = labeled.sort_values([id_col, "_year"]).copy()
    rows["_span_group"] = rows.groupby(id_col, sort=False)[LABEL_COL].transform(lambda s: s.ne(s.shift()).cumsum())
    return _aggregate_spans(rows, id_col, target_col, feature_names)


def build_transition_spans(
    labeled: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    feature_names: list[str],
) -> pd.DataFrame:
    rows = labeled[labeled["next_year"].notna()].sort_values([id_col, "_year"]).copy()
    rows["_transition_end_year"] = rows["next_year"]
    rows["_span_group"] = rows.groupby(id_col, sort=False)[LABEL_COL].transform(lambda s: s.ne(s.shift()).cumsum())
    return _aggregate_spans(rows, id_col, target_col, feature_names, end_col="_transition_end_year")


def _aggregate_spans(
    rows: pd.DataFrame,
    id_col: str,
    target_col: str,
    feature_names: list[str],
    end_col: str = "_year",
) -> pd.DataFrame:
    year_agg = ["first", "last"] if end_col == "_year" else "first"
    agg = {
        "_year": year_agg,
        LABEL_COL: "last",
        target_col: ["mean", "first", "last"],
        "interesting_score": ["mean", "max"],
        "state": "last",
        "transition": lambda s: "; ".join(sorted(set(str(x) for x in s if str(x) and str(x) != "nan")))[:500],
    }
    for transition_col in ["transition_delta", "transition_residual", "transition_z", "transition_score"]:
        if transition_col in rows.columns:
            agg[transition_col] = "mean"
    if end_col != "_year":
        agg[end_col] = "last"
    for col in feature_names:
        if col in rows.columns:
            agg[col] = "last"

    spans = rows.groupby([id_col, "_span_group"], sort=False).agg(agg)
    flat_cols = []
    for col in spans.columns.to_flat_index():
        parts = [str(part) for part in col if str(part)]
        if len(parts) == 1:
            flat_cols.append(parts[0])
        else:
            flat_cols.append("_".join(parts).replace("<lambda_0>", "set").replace("<lambda>", "set"))
    spans.columns = flat_cols
    spans = spans.reset_index(drop=False)
    spans = spans.rename(
        columns={
            "_year_first": "start_year",
            "_year": "start_year",
            f"{end_col}_last": "end_year",
            end_col: "end_year",
            f"{LABEL_COL}_last": LABEL_COL,
            LABEL_COL: LABEL_COL,
            f"{target_col}_mean": "target_mean",
            f"{target_col}_first": "target_start",
            f"{target_col}_last": "target_end",
            "interesting_score_mean": "score_mean",
            "interesting_score_max": "score_max",
            "state_last": "end_state",
            "transition_set": "transitions",
            "transition_delta_mean": "transition_delta",
            "transition_residual_mean": "transition_residual",
            "transition_z_mean": "transition_z",
            "transition_score_mean": "transition_score",
        }
    )
    spans["span_length"] = (spans["end_year"] - spans["start_year"] + 1).astype(int)
    start = pd.to_numeric(spans.get("target_start"), errors="coerce")
    end = pd.to_numeric(spans.get("target_end"), errors="coerce")
    spans["target_pct_change"] = ((end - start) / start.replace(0, pd.NA)) * 100.0
    spans["span_id"] = (
        spans[id_col].astype(str)
        + "_"
        + spans["start_year"].astype(int).astype(str)
        + "_"
        + spans["end_year"].astype(int).astype(str)
        + "_"
        + spans[LABEL_COL].astype(int).astype(str)
    )
    spans = spans.rename(columns={f"{c}_last": c for c in feature_names})
    return spans.drop(columns=["_span_group"], errors="ignore")
