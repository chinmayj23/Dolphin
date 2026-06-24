from __future__ import annotations

import numpy as np
import pandas as pd


def extract_time_index(values: pd.Series, time_unit: str = "year") -> pd.Series:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.notna().any() and time_unit == "year":
        return dates.dt.year
    if dates.notna().any() and time_unit == "month":
        return dates.dt.to_period("M").astype("int64")
    return pd.to_numeric(values, errors="coerce")


def apply_target(df: pd.DataFrame, target_cfg: dict) -> tuple[pd.DataFrame, str, list[str]]:
    out = df.copy()
    mode = target_cfg.get("mode", "column")
    if mode == "column":
        return out, target_cfg["column"], [target_cfg["column"]]
    if mode == "ratio":
        output_col = target_cfg.get("output_col", f"{target_cfg['numerator']}_per_{target_cfg['denominator']}")
        numerator = pd.to_numeric(out[target_cfg["numerator"]], errors="coerce")
        denominator = pd.to_numeric(out[target_cfg["denominator"]], errors="coerce").replace(0, np.nan)
        out[output_col] = numerator / denominator
        return out, output_col, [output_col, target_cfg["numerator"], target_cfg["denominator"]]
    raise ValueError(f"Unknown target mode: {mode}")


def build_temporal_features(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    feature_cfg: dict,
    exclude_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    work = df.copy()
    time_unit = str(feature_cfg.get("time_unit", "year"))
    work["_year"] = extract_time_index(work[date_col], time_unit=time_unit)
    work = work.dropna(subset=[id_col, "_year", target_col]).sort_values([id_col, "_year"]).reset_index(drop=True)

    include = set(feature_cfg.get("include_columns", []))
    include_prefixes = tuple(str(x) for x in feature_cfg.get("include_prefixes", []))
    static = set(feature_cfg.get("static_columns", []))
    static_prefixes = tuple(str(x) for x in feature_cfg.get("static_prefixes", []))
    exclude = set(exclude_cols)
    base_cols = []
    for col in work.columns:
        if col in {id_col, date_col, "_year"} or col in exclude:
            continue
        if include and col not in include and not any(col.startswith(prefix) for prefix in include_prefixes):
            continue
        vals = pd.to_numeric(work[col], errors="coerce")
        if vals.notna().any():
            work[col] = vals
            base_cols.append(col)

    lags = [int(x) for x in feature_cfg.get("lags", [1, 3])]
    windows = [int(x) for x in feature_cfg.get("windows", [3, 5])]
    new_cols = {}
    feature_names = []
    for col in base_cols:
        values = pd.to_numeric(work[col], errors="coerce")
        latest = f"latest_{col}"
        new_cols[latest] = values
        feature_names.append(latest)
        if col in static or any(col.startswith(prefix) for prefix in static_prefixes):
            continue
        by_id = values.groupby(work[id_col], sort=False)
        for lag in lags:
            lag_col = f"lag{lag}_{col}"
            delta_col = f"delta{lag}_{col}"
            lag_values = by_id.shift(lag)
            new_cols[lag_col] = lag_values
            new_cols[delta_col] = values - lag_values
            feature_names.extend([lag_col, delta_col])
        for window in windows:
            roll = by_id.rolling(window=window, min_periods=2)
            mean_col = f"roll_mean{window}_{col}"
            std_col = f"roll_std{window}_{col}"
            new_cols[mean_col] = roll.mean().reset_index(level=0, drop=True)
            new_cols[std_col] = roll.std().reset_index(level=0, drop=True)
            feature_names.extend([mean_col, std_col])

    if new_cols:
        work = pd.concat([work, pd.DataFrame(new_cols, index=work.index)], axis=1)
    return work, feature_names
