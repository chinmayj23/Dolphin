# DOLPHIN

DOLPHIN discovers interpretable subgroups in longitudinal data.

Given a panel dataset, DOLPHIN compares each entity's target trajectory with the population trajectory baseline. It then learns simple rules over explanatory variables that describe entities with distinct trajectory shapes.

This repository includes a World Bank GDP-per-capita example.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run the World Bank Example

```powershell
python scripts/run_dolphin.py --config configs/world_bank_gdp.json --workspace-root .
```

The run uses:

```text
data/world_bank_development_indicators.csv
configs/world_bank_gdp.json
```

Outputs are written to:

```text
outputs/world_bank_gdp/gdp_per_capita/dolphin/
```

## Main Results

After the run, the most useful files are:

```text
rules.csv
rules_natural_language.txt
metrics.json
performance_vs_questions.pdf
plots/rule_XX_trajectory.pdf
forest_trees/tree_XX_idx_YYY.pdf
```

The curated result files used for the World Bank case study are also available in:

```text
artifacts/world_bank_gdp/
```

## What the Method Does

1. Builds one target trajectory per entity.
2. Mean-centers each trajectory to focus on shape rather than level.
3. Computes a global mean-centered trajectory baseline.
4. Scores each entity by its distance from that baseline.
5. Builds lag, window-mean, change, and volatility features from explanatory variables.
6. Trains compact random-forest surrogate models using the explanatory variables.
7. Selects a forest that balances separation quality and rule complexity.
8. Converts tree paths into readable subgroup rules.
9. Ranks rules by subgroup size, trajectory divergence, concentration, and diversity.

## Example Output

A rule has the form:

```text
IF 10-year mean birth rate > 21.15
AND 26-year mean internet use <= 15.10
THEN the matching countries follow a GDP-per-capita trajectory that differs from the global baseline.
```

The trajectory plots show the subgroup trend against the global mean-centered baseline. The forest plots show the questions used to form the rules.

Open the trajectory figures here:

```text
artifacts/world_bank_gdp/trajectories/rule_01_trajectory.pdf
artifacts/world_bank_gdp/trajectories/rule_02_trajectory.pdf
artifacts/world_bank_gdp/trajectories/rule_03_trajectory.pdf
artifacts/world_bank_gdp/trajectories/rule_04_trajectory.pdf
```
