from __future__ import annotations

from pathlib import Path

import pandas as pd


HISTORICAL_WORLD_BANK_AGGREGATES = {
    "East Asia & Pacific (IDA & IBRD)",
    "Europe & Central Asia (IDA & IBRD)",
    "Latin America & Caribbean",
    "Latin America & Caribbean (IDA & IBRD)",
    "Middle East & North Africa",
    "Middle East & North Africa (IDA & IBRD countries)",
    "Middle East & North Africa (IDA & IBRD)",
    "Middle East & North Africa (excluding high income)",
    "Sub-Saharan Africa",
    "Sub-Saharan Africa (IDA & IBRD)",
}


def preprocess_data(df: pd.DataFrame, data_cfg: dict, root: Path) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    report = {
        "input_rows": int(len(out)),
        "input_entities": int(out[data_cfg["id_col"]].nunique(dropna=True)),
        "aggregate_rows_removed": 0,
        "aggregate_entities_removed": 0,
    }
    if not data_cfg.get("remove_world_bank_aggregates", False):
        report["output_rows"] = report["input_rows"]
        report["output_entities"] = report["input_entities"]
        return out, report

    metadata_path = Path(data_cfg.get("world_bank_entity_metadata", ""))
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path
    if not metadata_path.exists():
        raise FileNotFoundError(f"World Bank entity metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    aggregate_flag = metadata["is_aggregate"].astype(str).str.lower().isin({"true", "1", "yes"})
    aggregate_names = set(metadata.loc[aggregate_flag, "name"].dropna().astype(str))
    aggregate_names.update(HISTORICAL_WORLD_BANK_AGGREGATES)
    entity_col = data_cfg["id_col"]
    remove_mask = out[entity_col].astype(str).isin(aggregate_names)
    report["aggregate_rows_removed"] = int(remove_mask.sum())
    report["aggregate_entities_removed"] = int(out.loc[remove_mask, entity_col].nunique(dropna=True))
    out = out.loc[~remove_mask].copy()
    report["output_rows"] = int(len(out))
    report["output_entities"] = int(out[entity_col].nunique(dropna=True))
    report["metadata_path"] = str(metadata_path)
    return out, report
