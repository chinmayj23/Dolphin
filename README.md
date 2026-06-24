# DOLPHIN

**DOLPHIN** is a Python implementation of interpretable subgroup discovery for longitudinal data.

DOLPHIN finds rule-defined subgroups whose target trajectories differ from a population trajectory baseline. The target trajectory is used to measure longitudinal exceptionality. The rules are formed only from independent explanatory covariates.

The current implementation supports:

- continuous trajectory-based subgroup discovery,
- compact forest-based surrogate rule generation,
- multi-interval lag/window feature engineering for longitudinal covariates,
- rule-level trajectory diagnostics,
- paper-oriented trajectory and tree visualizations,
- EDA notebooks for the World Bank and CMIE case studies.

Internally, some modules and output folders still use the historical name `trajtrack`. Treat those as implementation names. The method name used in figures and documentation is **DOLPHIN**.

## Repository Layout

```text
DOLPHIN/
  configs/                  JSON experiment configs
  data/                     local input data, not tracked by Git
  notebooks/                runnable notebooks
  outputs/                  generated outputs, not tracked by Git
  scripts/                  command-line runners and utilities
  src/tsd/                  source code
  README.md
  requirements.txt
  pyproject.toml
```

## Installation

Create and activate a Python environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For editable local development:

```powershell
python -m pip install -e .
```

## Data

Data files are intentionally not committed to Git.

Place input CSVs under:

```text
data/
  world_bank_development_indicators.csv
  cmie_trajtrack_panel.csv
```

The World Bank config also uses:

```text
configs/world_bank_entities.csv
```

This metadata file is included because it is small and is used to remove World Bank aggregate regions.

### Expected World Bank Columns

The GDP config expects at least:

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

GDP per capita is computed inside the pipeline as:

```text
GDP_current_US / population
```

### Expected CMIE Columns

The CMIE config expects at least:

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
```

Optional static one-hot prefixes:

```text
dominant_source_*
income_quintile_*
```

## Run DOLPHIN

Run from the repository root.

### World Bank GDP Case Study

```powershell
python scripts/run_dolphin.py --config configs/world_bank_gdp.json --workspace-root .
```

Outputs are written to:

```text
outputs/world_bank_gdp/gdp_per_capita/trajtrack/
```

### CMIE Income Case Study

```powershell
python scripts/run_dolphin.py --config configs/cmie_income.json --workspace-root .
```

Outputs are written to:

```text
outputs/cmie_income/total_income/trajtrack/
```

## Main Outputs

Each run writes a complete local output directory:

```text
rules.csv                         selected subgroup rules and metrics
rules_natural_language.txt        readable rule explanations
membership.csv                    entity-by-rule membership matrix
metrics.json                      run-level metrics
plots/                            per-rule trajectory plots
forest_trees/                     selected surrogate forest trees
```

The paper-ready subset is under:

```text
paper_artifacts/world_bank_gdp/
paper_artifacts/cmie_income/
```

These folders intentionally contain only trajectories, clean forest trees, performance-vs-questions plots, rules, memberships, and metrics. KDE plots and distribution-level diagnostic plots are not included.

## EDA Notebooks

The EDA notebooks explain the datasets, target construction, temporal coverage, missingness, feature availability, trajectory diagnostics, and validation of discovered rules.

```text
notebooks/eda_world_bank_gdp_dolphin.ipynb
notebooks/eda_cmie_dolphin.ipynb
```

Regenerate the notebooks from source:

```powershell
python scripts/create_eda_notebooks.py
```

Execute a notebook:

```powershell
python -m jupyter nbconvert --to notebook --execute notebooks/eda_world_bank_gdp_dolphin.ipynb --inplace
python -m jupyter nbconvert --to notebook --execute notebooks/eda_cmie_dolphin.ipynb --inplace
```

## Regenerate Trajectory Plots Only

If rules and memberships already exist, regenerate trajectory plots without rerunning the full forest search:

```powershell
python scripts/regenerate_dolphin_trajectory_plots.py --config configs/world_bank_gdp.json --target gdp_per_capita --workspace-root .
python scripts/regenerate_dolphin_trajectory_plots.py --config configs/cmie_income.json --target total_income --workspace-root .
```

## Algorithm Summary

For each entity, DOLPHIN builds a target trajectory and mean-centers it to emphasize temporal shape. It computes a global mean-centered baseline trajectory. Each entity receives a continuous anomaly score based on its distance from this baseline.

DOLPHIN then trains many compact random-forest surrogate models using only explanatory covariates. It evaluates candidate forests by separation quality and complexity, extracts root-to-leaf paths as Boolean candidate rules, and scores each rule by how far the matched subgroup trajectory is from the global baseline.

Rule quality combines:

```text
trajectory divergence
subgroup size
within-subgroup concentration
overlap and trajectory-diversity filtering
```

The target trajectory is never used as a rule condition.

## Git Guidance

Pushing only this directory is enough if you want a clean standalone DOLPHIN repository.

Recommended:

```powershell
cd transition_subgroup_discovery
git init
git add README.md requirements.txt pyproject.toml .gitignore data/README.md
git add configs/world_bank_gdp.json configs/cmie_income.json configs/world_bank_entities.csv
git add src scripts notebooks paper_artifacts
git status
git commit -m "Initial DOLPHIN release"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Do not commit:

```text
data/
outputs/
*.csv data dumps
__pycache__/
.ipynb_checkpoints/
```

These are ignored by `.gitignore`.

## Notes

- CMIE data may be private or licensed. Keep it out of public Git.
- Generated figures and outputs can be archived separately if needed.
- The current implementation keeps `trajtrack` in some internal function and folder names for backward compatibility.
