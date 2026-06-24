# DOLPHIN Paper Artifacts

This directory contains the paper-facing DOLPHIN outputs for the two case studies.

Included:

- selected subgroup rules,
- natural-language rule descriptions,
- run metrics,
- forest selection summaries,
- performance-vs-questions plots,
- per-rule trajectory plots,
- clean per-tree forest visualizations.

Excluded:

- KDE plots,
- auxiliary distribution diagnostics,
- exploratory validation figures,
- CMIE household identifiers,
- CMIE row-level subgroup memberships,
- raw input data.

Case-study folders:

```text
world_bank_gdp/
cmie_income/
```

Use the full `outputs/` directory only for local debugging. Use this folder for figures and tables that are intended to be shared with the paper.

For CMIE, this folder contains only aggregate subgroup outputs. Do not commit household-level membership files or any table that lists `HH_ID` values.
