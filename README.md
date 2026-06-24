# DOLPHIN

**DOLPHIN** discovers interpretable subgroups in longitudinal data.

DOLPHIN represents each entity by a target trajectory, measures how unusual that trajectory is relative to a population baseline, and learns compact rule sets over explanatory covariates. Rule conditions use only explanatory covariates. They describe which entities follow distinct longitudinal patterns.

The repository contains the code and experiment assets for the DOLPHIN paper case studies.

## Method

DOLPHIN runs the following pipeline:

1. Build an entity-level target trajectory from panel data.
2. Mean-center each trajectory to focus on temporal shape.
3. Compute a global mean-centered trajectory baseline.
4. Score each entity by its distance from the baseline trajectory.
5. Engineer lag, rolling-window mean, change, and volatility features from explanatory covariates.
6. Train compact random-forest surrogate models from covariates to trajectory anomaly scores.
7. Select a forest by separation quality under a question-budget tolerance.
8. Extract root-to-leaf paths as Boolean subgroup rules.
9. Score each subgroup by trajectory divergence, subgroup size, concentration, and rule diversity.
10. Export rules, trajectories, forest trees, and forest-selection diagnostics.

## Repository Layout

```text
configs/       Experiment configs
data/          World Bank case-study data
notebooks/     EDA and run notebooks
scripts/       Command-line entry points
src/tsd/       DOLPHIN source code
artifacts/     Selected case-study outputs
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Case Studies

### World Bank GDP Per Capita

The World Bank case study uses country-year observations and discovers subgroups with distinct GDP-per-capita trajectories.

```powershell
python scripts/run_dolphin.py --config configs/world_bank_gdp.json --workspace-root .
```

Output directory:

```text
outputs/world_bank_gdp/gdp_per_capita/dolphin/
```

### CMIE Household Income

The CMIE case study uses household-month observations and discovers subgroups with distinct total-income trajectories.

```powershell
python scripts/run_dolphin.py --config configs/cmie_income.json --workspace-root .
```

Output directory:

```text
outputs/cmie_income/total_income/dolphin/
```

## Outputs

Each run writes:

```text
rules.csv
rules_natural_language.txt
metrics.json
membership.csv
forest_summary.csv
forest_selection.csv
performance_vs_questions.{png,svg,pdf}
plots/rule_XX_trajectory.{png,svg,pdf}
forest_trees/tree_XX_idx_YYY.{png,svg,pdf}
```

Selected case-study outputs are stored in:

```text
artifacts/world_bank_gdp/
artifacts/cmie_income/
```

## Notebooks

```text
notebooks/eda_world_bank_gdp_dolphin.ipynb
notebooks/eda_cmie_dolphin.ipynb
notebooks/run_cmie_dolphin.ipynb
notebooks/run_world_bank.ipynb
```

Regenerate the EDA notebooks:

```powershell
python scripts/create_eda_notebooks.py
```

Execute the EDA notebooks:

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks/eda_world_bank_gdp_dolphin.ipynb --inplace
python -m jupyter nbconvert --to notebook --execute notebooks/eda_cmie_dolphin.ipynb --inplace
```

## Plot Regeneration

Regenerate trajectory plots from existing rule and membership outputs:

```powershell
python scripts/regenerate_dolphin_trajectory_plots.py --config configs/world_bank_gdp.json --target gdp_per_capita --workspace-root .
python scripts/regenerate_dolphin_trajectory_plots.py --config configs/cmie_income.json --target total_income --workspace-root .
```

## Data Schema

World Bank config:

```text
country
date
GDP_current_US
population
access_to_electricity%
birth_rate
death_rate
government_expenditure_on_education%
government_health_expenditure%
gini_index
human_capital_index
individuals_using_internet%
inflation_annual%
population_density
rural_population
tax_revenue%
trade_in_services%
voice_and_accountability_estimate
```

CMIE config:

```text
HH_ID
MONTH_SLOT_DATE
TOT_INC
wage_share
biz_share
transfer_share
capital_income_share
TOT_N
EMPLOYED_N
MAX_EDU_LEVEL
dominant_source_*
income_quintile_*
```
