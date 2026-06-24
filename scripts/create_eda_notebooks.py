from __future__ import annotations

from pathlib import Path

import nbformat as nbf


REPO = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO / "notebooks"


def code_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source.strip() + "\n")


def markdown_cell(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source.strip() + "\n")


def make_notebook(title: str, cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [markdown_cell(f"# {title}"), *cells]
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
        },
    }
    return nb


COMMON_SETUP = r"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO = Path.cwd()
if not (REPO / "transition_subgroup_discovery").exists():
    for parent in Path.cwd().parents:
        if (parent / "transition_subgroup_discovery").exists():
            REPO = parent
            break

SRC = REPO / "transition_subgroup_discovery" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsd.features import apply_target, build_temporal_features
from tsd.io import ensure_dir, load_json
from tsd.preprocessing import preprocess_data

sns.set_theme(style="whitegrid", context="notebook")
pd.set_option("display.max_columns", 120)
pd.set_option("display.width", 160)
"""


def cmie_notebook() -> nbf.NotebookNode:
    cells = [
        markdown_cell(
            "This notebook is a self-contained audit of the CMIE panel used by DOLPHIN. "
            "It explains the panel structure, the target variable, the explanatory variables, "
            "the amount of temporal coverage, and the trajectory diagnostics used to validate "
            "the discovered subgroups. A reader who has not seen the raw data should be able "
            "to understand what the dataset contains and why it is suitable for longitudinal "
            "subgroup discovery."
        ),
        code_cell(COMMON_SETUP),
        markdown_cell(
            "## Dataset Role\n\n"
            "Each row is a household-month observation. The entity is the household, identified "
            "by `HH_ID`. The time variable is `MONTH_SLOT_DATE`. The target is `TOT_INC`, which "
            "is total household income in that month. DOLPHIN searches for rules over household "
            "covariates and income-composition variables, then checks whether the households "
            "satisfying each rule have unusual total-income trajectories."
        ),
        code_cell(
            r"""
CONFIG_PATH = REPO / "transition_subgroup_discovery" / "configs" / "cmie_dense_intervals.json"
cfg = load_json(CONFIG_PATH)
OUT_DIR = ensure_dir(REPO / "transition_subgroup_discovery" / "outputs" / "cmie_dense_intervals" / "eda")

data_path = REPO / cfg["data"]["path"]
raw = pd.read_csv(data_path)
raw["MONTH_SLOT_DATE"] = pd.to_datetime(raw["MONTH_SLOT_DATE"], errors="coerce")
raw["TOT_INC"] = pd.to_numeric(raw["TOT_INC"], errors="coerce")

print("Data path:", data_path)
print("Rows:", len(raw))
print("Households:", raw["HH_ID"].nunique())
print("Date range:", raw["MONTH_SLOT_DATE"].min(), "to", raw["MONTH_SLOT_DATE"].max())
print("Monthly periods:", raw["MONTH_SLOT_DATE"].nunique())
print("Output directory:", OUT_DIR)
raw.head()
"""
        ),
        markdown_cell("## Variable Dictionary"),
        code_cell(
            r"""
variable_dictionary = pd.DataFrame(
    [
        ("HH_ID", "Entity identifier", "Household ID used to link monthly observations into a trajectory."),
        ("MONTH_SLOT_DATE", "Time index", "Monthly observation date."),
        ("TOT_INC", "Target", "Total household income. This is the longitudinal target whose trajectory DOLPHIN evaluates."),
        ("wage_share", "Rule covariate", "Share of total income coming from wages."),
        ("biz_share", "Rule covariate", "Share of total income coming from business income."),
        ("transfer_share", "Rule covariate", "Share of total income coming from transfers."),
        ("capital_income_share", "Rule covariate", "Share of total income coming from capital income."),
        ("TOT_N", "Rule covariate", "Household size."),
        ("EMPLOYED_N", "Rule covariate", "Number of employed household members."),
        ("MAX_EDU_LEVEL", "Rule covariate", "Ordinal encoding of maximum education level in the household."),
        ("dominant_source_*", "Static rule covariate", "One-hot indicators for the household's dominant income source category."),
    ],
    columns=["column", "role", "meaning"],
)
variable_dictionary.to_csv(OUT_DIR / "cmie_variable_dictionary.csv", index=False)
variable_dictionary
"""
        ),
        markdown_cell(
            "The first table gives a compact dataset summary. The target is skewed, so the "
            "median and upper quantiles are more informative than the mean alone."
        ),
        code_cell(
            r"""
summary = pd.DataFrame({
    "metric": [
        "rows",
        "households",
        "monthly_periods",
        "target_nonmissing",
        "target_mean",
        "target_median",
        "target_p90",
        "target_p99",
    ],
    "value": [
        len(raw),
        raw["HH_ID"].nunique(),
        raw["MONTH_SLOT_DATE"].nunique(),
        raw["TOT_INC"].notna().sum(),
        raw["TOT_INC"].mean(),
        raw["TOT_INC"].median(),
        raw["TOT_INC"].quantile(0.90),
        raw["TOT_INC"].quantile(0.99),
    ],
})
summary.to_csv(OUT_DIR / "cmie_panel_summary.csv", index=False)
summary
"""
        ),
        markdown_cell("## Panel Coverage"),
        markdown_cell(
            "A longitudinal method needs repeated observations for the same entity. The next "
            "tables and plots show how many months are available per household and how many "
            "households are observed in each month. Large drops in either plot would indicate "
            "that trajectory estimates may be driven by sparse panel coverage."
        ),
        code_cell(
            r"""
coverage = raw.groupby("HH_ID").agg(
    n_months=("MONTH_SLOT_DATE", "nunique"),
    first_month=("MONTH_SLOT_DATE", "min"),
    last_month=("MONTH_SLOT_DATE", "max"),
    target_nonmissing=("TOT_INC", lambda s: s.notna().sum()),
    mean_income=("TOT_INC", "mean"),
    start_income=("TOT_INC", "first"),
    end_income=("TOT_INC", "last"),
)
coverage["target_change"] = coverage["end_income"] - coverage["start_income"]
coverage.to_csv(OUT_DIR / "cmie_entity_coverage.csv")
coverage.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
"""
        ),
        code_cell(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(coverage["n_months"], bins=40, ax=axes[0], color="#2878B5")
axes[0].axvline(coverage["n_months"].median(), color="black", linestyle="--", linewidth=1.2)
axes[0].set_title("Observed Months per Household")
axes[0].set_xlabel("Number of monthly observations")

monthly_counts = raw.groupby("MONTH_SLOT_DATE")["HH_ID"].nunique()
monthly_counts.plot(ax=axes[1], color="#D55E00", linewidth=2)
axes[1].set_title("Households Observed by Month")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Households")
fig.tight_layout()
fig.savefig(OUT_DIR / "cmie_panel_coverage.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        code_cell(
            r"""
coverage_quantiles = coverage["n_months"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_frame("observed_months")
coverage_quantiles.to_csv(OUT_DIR / "cmie_panel_coverage_quantiles.csv")
coverage_quantiles
"""
        ),
        markdown_cell("## Target Distribution and Temporal Pattern"),
        markdown_cell(
            "`TOT_INC` is the target trajectory. The histogram shows the cross-sectional "
            "income distribution after clipping the most extreme 1% for readability. The time "
            "series plot shows whether the target has a population-level drift over the panel."
        ),
        code_cell(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(raw["TOT_INC"].clip(upper=raw["TOT_INC"].quantile(0.99)), bins=60, ax=axes[0], color="#009E73")
axes[0].set_title("Total Income Distribution, Clipped at p99")
axes[0].set_xlabel("Total household income")

monthly_target = raw.groupby("MONTH_SLOT_DATE")["TOT_INC"].agg(["median", "mean", "count"])
monthly_target[["median", "mean"]].plot(ax=axes[1], linewidth=2)
axes[1].set_title("Monthly Total Income")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Income")
fig.tight_layout()
fig.savefig(OUT_DIR / "cmie_target_distribution_and_time.png", dpi=300, bbox_inches="tight")
plt.show()

monthly_target.to_csv(OUT_DIR / "cmie_monthly_target_summary.csv")
monthly_target.tail()
"""
        ),
        markdown_cell("## Example Household Trajectories"),
        code_cell(
            r"""
eligible = coverage.loc[coverage["n_months"] >= coverage["n_months"].quantile(0.75)].copy()
examples = eligible.sort_values("mean_income").iloc[
    np.linspace(0, max(len(eligible) - 1, 0), num=min(8, len(eligible)), dtype=int)
].index
example_panel = raw[raw["HH_ID"].isin(examples)].copy()

fig, ax = plt.subplots(figsize=(12, 6))
for hh_id, group in example_panel.groupby("HH_ID"):
    group = group.sort_values("MONTH_SLOT_DATE")
    ax.plot(group["MONTH_SLOT_DATE"], group["TOT_INC"], marker="o", linewidth=1.5, alpha=0.8, label=str(hh_id))
ax.set_title("Example Household Total-Income Trajectories")
ax.set_xlabel("Month")
ax.set_ylabel("Total household income")
ax.legend(title="HH_ID", fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(OUT_DIR / "cmie_example_household_trajectories.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("## Covariates Used by DOLPHIN"),
        markdown_cell(
            "The rule language uses income composition and household characteristics. The next "
            "outputs show missingness and marginal distributions for the main covariates. This "
            "helps identify variables that may dominate rules because of scale, sparsity, or "
            "strong concentration near zero."
        ),
        code_cell(
            r"""
feature_cols = [
    "wage_share",
    "biz_share",
    "transfer_share",
    "capital_income_share",
    "EMPLOYED_N",
    "TOT_N",
    "MAX_EDU_LEVEL",
]
feature_cols = [c for c in feature_cols if c in raw.columns]
missing = raw[feature_cols].isna().mean().sort_values(ascending=False).to_frame("missing_fraction")
missing.to_csv(OUT_DIR / "cmie_covariate_missingness.csv")
missing
"""
        ),
        code_cell(
            r"""
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plot_cols = [c for c in ["wage_share", "biz_share", "transfer_share", "capital_income_share"] if c in raw.columns]
for ax, col in zip(axes.ravel(), plot_cols):
    sns.histplot(raw[col].dropna(), bins=40, ax=ax)
    ax.set_title(col.replace("_", " "))
for ax in axes.ravel()[len(plot_cols):]:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "cmie_income_share_distributions.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        code_cell(
            r"""
categorical_like = [c for c in raw.columns if c.startswith("dominant_source_") or c.startswith("income_quintile_")]
if categorical_like:
    static_rates = raw.groupby("HH_ID")[categorical_like].max().mean().sort_values(ascending=False).to_frame("share_of_households")
    static_rates.to_csv(OUT_DIR / "cmie_static_indicator_distribution.csv")
    display(static_rates)
else:
    print("No static one-hot categorical indicators found.")
"""
        ),
        markdown_cell("## Missingness by Month"),
        code_cell(
            r"""
monthly_missing = raw.groupby("MONTH_SLOT_DATE")[feature_cols + ["TOT_INC"]].apply(lambda frame: frame.isna().mean())
monthly_missing.to_csv(OUT_DIR / "cmie_monthly_missingness.csv")

fig, ax = plt.subplots(figsize=(12, 5))
sns.heatmap(monthly_missing.T, cmap="viridis", cbar_kws={"label": "Missing fraction"}, ax=ax)
ax.set_title("CMIE Missingness by Month")
ax.set_xlabel("Month")
ax.set_ylabel("Variable")
ax.set_xticks(np.linspace(0, len(monthly_missing.index) - 1, min(8, len(monthly_missing.index))).astype(int))
ax.set_xticklabels([monthly_missing.index[i].strftime("%Y-%m") for i in ax.get_xticks().astype(int)], rotation=35, ha="right")
fig.tight_layout()
fig.savefig(OUT_DIR / "cmie_monthly_missingness_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("## Dense Lag and Window Coverage"),
        markdown_cell(
            "The dense feature setting considers candidate lags and rolling windows every two "
            "months. We keep intervals only when at least 80% of households have enough target "
            "history. This avoids building rule features at horizons that only a minority of "
            "households can support."
        ),
        code_cell(
            r"""
counts = raw.dropna(subset=["TOT_INC"]).groupby("HH_ID")["MONTH_SLOT_DATE"].nunique()
interval_rows = []
for k in range(2, 41, 2):
    interval_rows.append({
        "interval_months": k,
        "entity_target_coverage": (counts >= k + 1).mean(),
        "passes_80pct": (counts >= k + 1).mean() >= 0.8,
    })
interval_coverage = pd.DataFrame(interval_rows)
interval_coverage.to_csv(OUT_DIR / "cmie_dense_interval_coverage.csv", index=False)
interval_coverage
"""
        ),
        code_cell(
            r"""
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(interval_coverage["interval_months"], interval_coverage["entity_target_coverage"], marker="o")
ax.axhline(0.8, color="black", linestyle="--", linewidth=1.2)
ax.set_ylim(0, 1.05)
ax.set_title("CMIE Target-History Coverage by Interval")
ax.set_xlabel("Lag/window length in months")
ax.set_ylabel("Share of households with enough target history")
fig.tight_layout()
fig.savefig(OUT_DIR / "cmie_dense_interval_coverage.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("## DOLPHIN Feature Matrix Availability"),
        markdown_cell(
            "After temporal feature engineering, DOLPHIN keeps only engineered features with at "
            "least 80% availability at the entity level. This section reports how many candidate "
            "features are created and how many survive that filter."
        ),
        code_cell(
            r"""
work, target_col, target_excludes = apply_target(raw, cfg["targets"][0])
table, feature_names = build_temporal_features(
    work,
    id_col=cfg["data"]["id_col"],
    date_col=cfg["data"]["date_col"],
    target_col=target_col,
    feature_cfg=cfg["feature_engineering"],
    exclude_cols=target_excludes,
)
entity_table = table.groupby(cfg["data"]["id_col"], sort=False).tail(1)
X = entity_table[feature_names].apply(pd.to_numeric, errors="coerce")
feature_missing = X.isna().mean().sort_values()
feature_report = pd.DataFrame({
    "feature": feature_missing.index,
    "missing_fraction": feature_missing.values,
    "retained_at_80pct": feature_missing.values < cfg["methods"]["trajtrack"]["max_missing_frac"],
})
feature_report.to_csv(OUT_DIR / "cmie_engineered_feature_availability.csv", index=False)
print("Engineered features:", len(feature_names))
print("Retained at 80% availability:", int(feature_report["retained_at_80pct"].sum()))
feature_report.head(20)
"""
        ),
        markdown_cell("## Trajectory Diagnostics"),
        markdown_cell(
            "These diagnostics summarize the target trajectories before subgroup discovery. "
            "`baseline_deviation` measures how far a household's mean-centered trajectory is "
            "from the global mean-centered trajectory. `mean_abs_monthly_change` measures how "
            "much the household's income changes from month to month on average."
        ),
        code_cell(
            r"""
pivot = raw.pivot_table(index="HH_ID", columns="MONTH_SLOT_DATE", values="TOT_INC", aggfunc="mean")
valid = pivot.dropna(thresh=6)
interp = valid.interpolate(axis=1, limit_direction="both")
values = interp.to_numpy(dtype=float)
centered = values - values.mean(axis=1, keepdims=True)
baseline = centered.mean(axis=0)
baseline_deviation = np.sqrt(((centered - baseline) ** 2).sum(axis=1))
mean_abs_change = np.mean(np.abs(np.diff(values, axis=1)), axis=1)
diag = pd.DataFrame({
    "HH_ID": interp.index,
    "baseline_deviation": baseline_deviation,
    "mean_abs_monthly_change": mean_abs_change,
    "mean_income": values.mean(axis=1),
    "start_income": values[:, 0],
    "end_income": values[:, -1],
})
diag.to_csv(OUT_DIR / "cmie_trajectory_diagnostics.csv", index=False)
diag.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
"""
        ),
        code_cell(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(diag["baseline_deviation"].clip(upper=diag["baseline_deviation"].quantile(0.99)), bins=60, ax=axes[0], color="#CC79A7")
axes[0].set_title("Trajectory Deviation from Global Baseline")
axes[0].set_xlabel("Euclidean deviation, clipped at p99")

sns.histplot(diag["mean_abs_monthly_change"].clip(upper=diag["mean_abs_monthly_change"].quantile(0.99)), bins=60, ax=axes[1], color="#0072B2")
axes[1].set_title("Mean Absolute Month-to-Month Income Change")
axes[1].set_xlabel("Mean absolute monthly change, clipped at p99")
fig.tight_layout()
fig.savefig(OUT_DIR / "cmie_trajectory_diagnostics.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("## Result Validation: Do DOLPHIN Rules Select Distinct Trajectories?"),
        markdown_cell(
            "This section links the EDA back to the discovered DOLPHIN rules. It checks whether "
            "the selected subgroups differ from the full population on trajectory diagnostics. "
            "A useful rule should generally show higher trajectory deviation, higher dynamic "
            "change, or a clearly different mean-centered trajectory shape."
        ),
        code_cell(
            r"""
RESULT_DIR = REPO / "transition_subgroup_discovery" / "outputs" / "cmie_dense_intervals" / "total_income" / "trajtrack"
rules = pd.read_csv(RESULT_DIR / "rules.csv")
membership = pd.read_csv(RESULT_DIR / "membership.csv")

display_cols = [
    "rank",
    "rule",
    "n_entities",
    "support",
    "baseline_effect_size",
    "subgroup_trend_change",
    "trend_change_difference",
    "relative_trend_direction",
]
rules[display_cols].to_csv(OUT_DIR / "cmie_dolphin_rule_validation_table.csv", index=False)
rules[display_cols]
"""
        ),
        code_cell(
            r"""
diag_valid = diag.merge(membership, left_on="HH_ID", right_on="entity", how="left")
rule_cols = [c for c in membership.columns if c.startswith("rule_")]
records = []
for col in rule_cols:
    rank = int(col.split("_")[1])
    mask = diag_valid[col].fillna(0).astype(bool)
    complement = ~mask
    for metric in ["baseline_deviation", "mean_abs_monthly_change"]:
        records.append({
            "rule": f"Rule {rank}",
            "metric": metric,
            "subgroup_median": diag_valid.loc[mask, metric].median(),
            "population_median": diag_valid[metric].median(),
            "complement_median": diag_valid.loc[complement, metric].median(),
            "median_ratio_to_population": diag_valid.loc[mask, metric].median() / max(diag_valid[metric].median(), 1e-12),
            "n": int(mask.sum()),
        })
validation_summary = pd.DataFrame(records)
validation_summary.to_csv(OUT_DIR / "cmie_dolphin_subgroup_validation_summary.csv", index=False)
validation_summary
"""
        ),
        code_cell(
            r"""
plot_rows = []
for col in rule_cols:
    rank = int(col.split("_")[1])
    mask = diag_valid[col].fillna(0).astype(bool)
    subgroup = diag_valid.loc[mask, ["baseline_deviation", "mean_abs_monthly_change"]].copy()
    subgroup["group"] = f"Rule {rank}"
    plot_rows.append(subgroup)
population_sample = diag_valid[["baseline_deviation", "mean_abs_monthly_change"]].sample(
    n=min(3000, len(diag_valid)), random_state=42
).copy()
population_sample["group"] = "Population"
plot_df = pd.concat([population_sample, *plot_rows], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(
    data=plot_df,
    x="group",
    y="baseline_deviation",
    ax=axes[0],
    showfliers=False,
    color="#8DBBD9",
)
axes[0].set_title("Baseline-Deviation Validation")
axes[0].set_xlabel("")
axes[0].set_ylabel("Deviation from global trajectory baseline")
axes[0].tick_params(axis="x", rotation=25)

sns.boxplot(
    data=plot_df,
    x="group",
    y="mean_abs_monthly_change",
    ax=axes[1],
    showfliers=False,
    color="#E6A57E",
)
axes[1].set_title("Trajectory-Change Validation")
axes[1].set_xlabel("")
axes[1].set_ylabel("Mean absolute month-to-month change")
axes[1].tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(OUT_DIR / "cmie_dolphin_subgroup_validation_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        code_cell(
            r"""
rule_colors = sns.color_palette("tab10", n_colors=max(3, len(rule_cols)))
fig, ax = plt.subplots(figsize=(11, 6))
time_axis = np.arange(values.shape[1])
iqr_low = np.quantile(centered, 0.25, axis=0)
iqr_high = np.quantile(centered, 0.75, axis=0)
ax.fill_between(time_axis, iqr_low, iqr_high, color="lightgray", alpha=0.65, label="Population IQR")
ax.plot(time_axis, baseline, color="black", linewidth=2.2, label="Global baseline")
for idx, col in enumerate(rule_cols):
    rank = int(col.split("_")[1])
    member_ids = set(membership.loc[membership[col] == 1, "entity"])
    mask = np.array([hh in member_ids for hh in interp.index])
    if mask.sum() == 0:
        continue
    ax.plot(time_axis, centered[mask].mean(axis=0), linewidth=2.4, color=rule_colors[idx], label=f"Rule {rank} (n={mask.sum()})")
ax.set_title("DOLPHIN Rule Trajectories on CMIE Total Income")
ax.set_xlabel("Aligned trajectory grid")
ax.set_ylabel("Mean-centered total income")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "cmie_dolphin_rule_trajectory_overlay.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
    ]
    return make_notebook("CMIE EDA for DOLPHIN", cells)


def world_bank_notebook() -> nbf.NotebookNode:
    cells = [
        markdown_cell(
            "This notebook is a self-contained audit of the World Bank GDP-per-capita case "
            "study used by DOLPHIN. It explains the entity-time structure, target construction, "
            "country coverage, covariate missingness, dense lag/window feasibility, and the "
            "trajectory diagnostics used to validate the discovered subgroups."
        ),
        code_cell(COMMON_SETUP),
        markdown_cell(
            "## Dataset Role\n\n"
            "Each row is a country-year observation. The entity is the country, identified by "
            "`country`. The time variable is `date`. The target is GDP per capita, computed as "
            "`GDP_current_US / population`. DOLPHIN searches for rules over development indicators "
            "and evaluates whether countries satisfying each rule have unusual GDP-per-capita "
            "trajectories."
        ),
        code_cell(
            r"""
CONFIG_PATH = REPO / "transition_subgroup_discovery" / "configs" / "world_bank_gdp_dense_intervals.json"
cfg = load_json(CONFIG_PATH)
OUT_DIR = ensure_dir(REPO / "transition_subgroup_discovery" / "outputs" / "world_bank_gdp_dense_intervals" / "eda")

data_path = REPO / cfg["data"]["path"]
raw = pd.read_csv(data_path)
raw, preprocess_report = preprocess_data(raw, cfg["data"], REPO)
raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
raw["GDP_current_US"] = pd.to_numeric(raw["GDP_current_US"], errors="coerce")
raw["population"] = pd.to_numeric(raw["population"], errors="coerce")
raw["gdp_per_capita"] = raw["GDP_current_US"] / raw["population"].replace(0, np.nan)

print("Data path:", data_path)
print("Rows after aggregate removal:", len(raw))
print("Countries:", raw["country"].nunique())
print("Date range:", raw["date"].min(), "to", raw["date"].max())
print("Annual periods:", raw["date"].nunique())
print("Preprocessing:", preprocess_report)
print("Output directory:", OUT_DIR)
raw.head()
"""
        ),
        markdown_cell("## Variable Dictionary"),
        code_cell(
            r"""
variable_dictionary = pd.DataFrame(
    [
        ("country", "Entity identifier", "Country name."),
        ("date", "Time index", "Annual observation date."),
        ("GDP_current_US", "Target component", "GDP in current US dollars."),
        ("population", "Target component", "Population. Used as denominator for GDP per capita."),
        ("gdp_per_capita", "Target", "GDP per capita, computed from GDP and population."),
        ("access_to_electricity%", "Rule covariate", "Infrastructure access indicator."),
        ("birth_rate", "Rule covariate", "Demographic indicator."),
        ("death_rate", "Rule covariate", "Demographic and health indicator."),
        ("government_expenditure_on_education%", "Rule covariate", "Education expenditure indicator."),
        ("government_health_expenditure%", "Rule covariate", "Health expenditure indicator."),
        ("gini_index", "Rule covariate", "Income inequality indicator."),
        ("human_capital_index", "Rule covariate", "Human capital indicator."),
        ("individuals_using_internet%", "Rule covariate", "Digital adoption indicator."),
        ("inflation_annual%", "Rule covariate", "Macroeconomic indicator."),
        ("population_density", "Rule covariate", "Demographic density indicator."),
        ("rural_population", "Rule covariate", "Rural population count."),
        ("tax_revenue%", "Rule covariate", "Public finance indicator."),
        ("trade_in_services%", "Rule covariate", "Service-trade intensity indicator."),
        ("voice_and_accountability_estimate", "Rule covariate", "Governance indicator."),
    ],
    columns=["column", "role", "meaning"],
)
variable_dictionary.to_csv(OUT_DIR / "world_bank_variable_dictionary.csv", index=False)
variable_dictionary
"""
        ),
        markdown_cell(
            "The summary below describes the post-filter country-level dataset after removing "
            "World Bank aggregate regions."
        ),
        code_cell(
            r"""
summary = pd.DataFrame({
    "metric": [
        "rows",
        "countries",
        "annual_periods",
        "gdp_pc_nonmissing",
        "gdp_pc_mean",
        "gdp_pc_median",
        "gdp_pc_p90",
        "gdp_pc_p99",
    ],
    "value": [
        len(raw),
        raw["country"].nunique(),
        raw["date"].nunique(),
        raw["gdp_per_capita"].notna().sum(),
        raw["gdp_per_capita"].mean(),
        raw["gdp_per_capita"].median(),
        raw["gdp_per_capita"].quantile(0.90),
        raw["gdp_per_capita"].quantile(0.99),
    ],
})
summary.to_csv(OUT_DIR / "world_bank_gdp_summary.csv", index=False)
summary
"""
        ),
        markdown_cell("## Country Coverage"),
        markdown_cell(
            "The country coverage plots show how much GDP-per-capita history is available per "
            "country and how many countries contribute target values in each year. This matters "
            "because DOLPHIN evaluates full country trajectories, not isolated annual records."
        ),
        code_cell(
            r"""
coverage = raw.groupby("country").agg(
    n_years=("date", "nunique"),
    n_gdp_pc=("gdp_per_capita", lambda s: s.notna().sum()),
    first_year=("date", "min"),
    last_year=("date", "max"),
    mean_gdp_pc=("gdp_per_capita", "mean"),
    start_gdp_pc=("gdp_per_capita", "first"),
    end_gdp_pc=("gdp_per_capita", "last"),
)
coverage["target_change"] = coverage["end_gdp_pc"] - coverage["start_gdp_pc"]
coverage.to_csv(OUT_DIR / "world_bank_country_coverage.csv")
coverage.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
"""
        ),
        code_cell(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(coverage["n_gdp_pc"], bins=35, ax=axes[0], color="#2878B5")
axes[0].axvline(coverage["n_gdp_pc"].median(), color="black", linestyle="--", linewidth=1.2)
axes[0].set_title("GDP-per-Capita Observations per Country")
axes[0].set_xlabel("Non-missing annual observations")

year_counts = raw.dropna(subset=["gdp_per_capita"]).groupby("date")["country"].nunique()
year_counts.plot(ax=axes[1], color="#D55E00", linewidth=2)
axes[1].set_title("Countries with GDP per Capita by Year")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Countries")
fig.tight_layout()
fig.savefig(OUT_DIR / "world_bank_gdp_coverage.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        code_cell(
            r"""
coverage_quantiles = coverage["n_gdp_pc"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_frame("nonmissing_gdp_pc_years")
coverage_quantiles.to_csv(OUT_DIR / "world_bank_country_coverage_quantiles.csv")
coverage_quantiles
"""
        ),
        markdown_cell("## Target Distribution and Global Trend"),
        markdown_cell(
            "GDP per capita is highly skewed across countries. The histogram is clipped at the "
            "99th percentile for readability. The time plot shows the population-level evolution "
            "of GDP per capita across countries."
        ),
        code_cell(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
gdp_clip = raw["gdp_per_capita"].clip(upper=raw["gdp_per_capita"].quantile(0.99))
sns.histplot(gdp_clip.dropna(), bins=60, ax=axes[0], color="#009E73")
axes[0].set_title("GDP per Capita Distribution, Clipped at p99")
axes[0].set_xlabel("GDP per capita")

yearly_target = raw.groupby("date")["gdp_per_capita"].agg(["median", "mean", "count"])
yearly_target[["median", "mean"]].plot(ax=axes[1], linewidth=2)
axes[1].set_title("World Bank GDP per Capita Over Time")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("GDP per capita")
fig.tight_layout()
fig.savefig(OUT_DIR / "world_bank_gdp_target_distribution_and_time.png", dpi=300, bbox_inches="tight")
plt.show()

yearly_target.to_csv(OUT_DIR / "world_bank_yearly_gdp_summary.csv")
yearly_target.tail()
"""
        ),
        markdown_cell("## Example Country Trajectories"),
        code_cell(
            r"""
eligible = coverage.loc[coverage["n_gdp_pc"] >= coverage["n_gdp_pc"].quantile(0.75)].copy()
examples = eligible.sort_values("mean_gdp_pc").iloc[
    np.linspace(0, max(len(eligible) - 1, 0), num=min(10, len(eligible)), dtype=int)
].index
example_panel = raw[raw["country"].isin(examples)].copy()

fig, ax = plt.subplots(figsize=(12, 6))
for country, group in example_panel.groupby("country"):
    group = group.sort_values("date")
    ax.plot(group["date"], group["gdp_per_capita"], linewidth=1.6, alpha=0.85, label=str(country))
ax.set_title("Example Country GDP-per-Capita Trajectories")
ax.set_xlabel("Year")
ax.set_ylabel("GDP per capita")
ax.legend(title="Country", fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(OUT_DIR / "world_bank_example_country_trajectories.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("## Covariate Missingness"),
        markdown_cell(
            "The rule language uses development indicators. The missingness plot shows which "
            "indicators are broadly available and which are sparse. Sparse indicators may still "
            "be useful, but they are less likely to survive the 80% engineered-feature availability "
            "filter."
        ),
        code_cell(
            r"""
covariates = cfg["feature_engineering"]["include_columns"]
missing = raw[covariates].isna().mean().sort_values(ascending=False).to_frame("missing_fraction")
missing.to_csv(OUT_DIR / "world_bank_covariate_missingness.csv")
missing
"""
        ),
        code_cell(
            r"""
fig, ax = plt.subplots(figsize=(8, 6))
missing.sort_values("missing_fraction").plot(kind="barh", legend=False, ax=ax, color="#CC79A7")
ax.set_title("World Bank Covariate Missingness")
ax.set_xlabel("Missing fraction")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(OUT_DIR / "world_bank_covariate_missingness.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("## Missingness by Year"),
        code_cell(
            r"""
yearly_missing = raw.groupby("date")[covariates + ["gdp_per_capita"]].apply(lambda frame: frame.isna().mean())
yearly_missing.to_csv(OUT_DIR / "world_bank_yearly_missingness.csv")

fig, ax = plt.subplots(figsize=(13, 6))
sns.heatmap(yearly_missing.T, cmap="viridis", cbar_kws={"label": "Missing fraction"}, ax=ax)
ax.set_title("World Bank Missingness by Year")
ax.set_xlabel("Year")
ax.set_ylabel("Variable")
tick_positions = np.linspace(0, len(yearly_missing.index) - 1, min(10, len(yearly_missing.index))).astype(int)
ax.set_xticks(tick_positions)
ax.set_xticklabels([yearly_missing.index[i].year for i in tick_positions], rotation=35, ha="right")
fig.tight_layout()
fig.savefig(OUT_DIR / "world_bank_yearly_missingness_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("## Dense Lag and Window Coverage"),
        markdown_cell(
            "Dense DOLPHIN features use lag and rolling-window lengths every two years. The "
            "coverage curve shows why the experiment keeps intervals up to 32 years: beyond "
            "that point, fewer than 80% of countries have enough target history."
        ),
        code_cell(
            r"""
counts = raw.dropna(subset=["gdp_per_capita"]).groupby("country")["date"].nunique()
interval_rows = []
for k in range(2, 65, 2):
    interval_rows.append({
        "interval_years": k,
        "entity_target_coverage": (counts >= k + 1).mean(),
        "passes_80pct": (counts >= k + 1).mean() >= 0.8,
    })
interval_coverage = pd.DataFrame(interval_rows)
interval_coverage.to_csv(OUT_DIR / "world_bank_dense_interval_coverage.csv", index=False)
interval_coverage
"""
        ),
        code_cell(
            r"""
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(interval_coverage["interval_years"], interval_coverage["entity_target_coverage"], marker="o")
ax.axhline(0.8, color="black", linestyle="--", linewidth=1.2)
ax.set_ylim(0, 1.05)
ax.set_title("World Bank GDP Target-History Coverage by Interval")
ax.set_xlabel("Lag/window length in years")
ax.set_ylabel("Share of countries with enough target history")
fig.tight_layout()
fig.savefig(OUT_DIR / "world_bank_dense_interval_coverage.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("## DOLPHIN Feature Matrix Availability"),
        markdown_cell(
            "This section constructs the same temporal feature matrix used by DOLPHIN and reports "
            "how many engineered features remain after the 80% availability filter."
        ),
        code_cell(
            r"""
work, target_col, target_excludes = apply_target(raw, cfg["targets"][0])
table, feature_names = build_temporal_features(
    work,
    id_col=cfg["data"]["id_col"],
    date_col=cfg["data"]["date_col"],
    target_col=target_col,
    feature_cfg=cfg["feature_engineering"],
    exclude_cols=target_excludes,
)
entity_table = table.groupby(cfg["data"]["id_col"], sort=False).tail(1)
X = entity_table[feature_names].apply(pd.to_numeric, errors="coerce")
feature_missing = X.isna().mean().sort_values()
feature_report = pd.DataFrame({
    "feature": feature_missing.index,
    "missing_fraction": feature_missing.values,
    "retained_at_80pct": feature_missing.values < cfg["methods"]["trajtrack"]["max_missing_frac"],
})
feature_report.to_csv(OUT_DIR / "world_bank_engineered_feature_availability.csv", index=False)
print("Engineered features:", len(feature_names))
print("Retained at 80% availability:", int(feature_report["retained_at_80pct"].sum()))
feature_report.head(20)
"""
        ),
        markdown_cell("## Trajectory Diagnostics"),
        markdown_cell(
            "These diagnostics summarize country GDP-per-capita trajectories before subgroup "
            "selection. `baseline_deviation` measures how far a country's mean-centered trajectory "
            "is from the global trajectory baseline. `mean_abs_annual_change` measures average "
            "year-to-year movement."
        ),
        code_cell(
            r"""
pivot = raw.pivot_table(index="country", columns="date", values="gdp_per_capita", aggfunc="mean")
valid = pivot.dropna(thresh=6)
interp = valid.interpolate(axis=1, limit_direction="both")
values = interp.to_numpy(dtype=float)
centered = values - values.mean(axis=1, keepdims=True)
baseline = centered.mean(axis=0)
baseline_deviation = np.sqrt(((centered - baseline) ** 2).sum(axis=1))
mean_abs_change = np.mean(np.abs(np.diff(values, axis=1)), axis=1)
diag = pd.DataFrame({
    "country": interp.index,
    "baseline_deviation": baseline_deviation,
    "mean_abs_annual_change": mean_abs_change,
    "mean_gdp_per_capita": values.mean(axis=1),
    "start_gdp_per_capita": values[:, 0],
    "end_gdp_per_capita": values[:, -1],
})
diag.to_csv(OUT_DIR / "world_bank_trajectory_diagnostics.csv", index=False)
diag.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
"""
        ),
        code_cell(
            r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
sns.histplot(diag["baseline_deviation"].clip(upper=diag["baseline_deviation"].quantile(0.99)), bins=50, ax=axes[0], color="#CC79A7")
axes[0].set_title("Trajectory Deviation from Global Baseline")
axes[0].set_xlabel("Euclidean deviation, clipped at p99")

sns.histplot(diag["mean_abs_annual_change"].clip(upper=diag["mean_abs_annual_change"].quantile(0.99)), bins=50, ax=axes[1], color="#0072B2")
axes[1].set_title("Mean Absolute Year-to-Year GDP-per-Capita Change")
axes[1].set_xlabel("Mean absolute annual change, clipped at p99")
fig.tight_layout()
fig.savefig(OUT_DIR / "world_bank_trajectory_diagnostics.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        markdown_cell("## Result Validation: Do DOLPHIN Rules Select Distinct Trajectories?"),
        markdown_cell(
            "This section checks whether the discovered DOLPHIN rules select countries whose "
            "trajectory diagnostics differ from the population. It is a validation step rather "
            "than a rule-search step."
        ),
        code_cell(
            r"""
RESULT_DIR = REPO / "transition_subgroup_discovery" / "outputs" / "world_bank_gdp_dense_intervals" / "gdp_per_capita" / "trajtrack"
rules = pd.read_csv(RESULT_DIR / "rules.csv")
membership = pd.read_csv(RESULT_DIR / "membership.csv")

display_cols = [
    "rank",
    "rule",
    "n_entities",
    "support",
    "baseline_effect_size",
    "subgroup_trend_change",
    "trend_change_difference",
    "relative_trend_direction",
]
rules[display_cols].to_csv(OUT_DIR / "world_bank_dolphin_rule_validation_table.csv", index=False)
rules[display_cols]
"""
        ),
        code_cell(
            r"""
diag_valid = diag.merge(membership, left_on="country", right_on="entity", how="left")
rule_cols = [c for c in membership.columns if c.startswith("rule_")]
records = []
for col in rule_cols:
    rank = int(col.split("_")[1])
    mask = diag_valid[col].fillna(0).astype(bool)
    complement = ~mask
    for metric in ["baseline_deviation", "mean_abs_annual_change"]:
        records.append({
            "rule": f"Rule {rank}",
            "metric": metric,
            "subgroup_median": diag_valid.loc[mask, metric].median(),
            "population_median": diag_valid[metric].median(),
            "complement_median": diag_valid.loc[complement, metric].median(),
            "median_ratio_to_population": diag_valid.loc[mask, metric].median() / max(diag_valid[metric].median(), 1e-12),
            "n": int(mask.sum()),
        })
validation_summary = pd.DataFrame(records)
validation_summary.to_csv(OUT_DIR / "world_bank_dolphin_subgroup_validation_summary.csv", index=False)
validation_summary
"""
        ),
        code_cell(
            r"""
plot_rows = []
for col in rule_cols:
    rank = int(col.split("_")[1])
    mask = diag_valid[col].fillna(0).astype(bool)
    subgroup = diag_valid.loc[mask, ["baseline_deviation", "mean_abs_annual_change"]].copy()
    subgroup["group"] = f"Rule {rank}"
    plot_rows.append(subgroup)
population_sample = diag_valid[["baseline_deviation", "mean_abs_annual_change"]].copy()
population_sample["group"] = "Population"
plot_df = pd.concat([population_sample, *plot_rows], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(
    data=plot_df,
    x="group",
    y="baseline_deviation",
    ax=axes[0],
    showfliers=False,
    color="#8DBBD9",
)
axes[0].set_title("Baseline-Deviation Validation")
axes[0].set_xlabel("")
axes[0].set_ylabel("Deviation from global trajectory baseline")
axes[0].tick_params(axis="x", rotation=25)

sns.boxplot(
    data=plot_df,
    x="group",
    y="mean_abs_annual_change",
    ax=axes[1],
    showfliers=False,
    color="#E6A57E",
)
axes[1].set_title("Trajectory-Change Validation")
axes[1].set_xlabel("")
axes[1].set_ylabel("Mean absolute year-to-year change")
axes[1].tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(OUT_DIR / "world_bank_dolphin_subgroup_validation_boxplots.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
        code_cell(
            r"""
rule_colors = sns.color_palette("tab10", n_colors=max(4, len(rule_cols)))
fig, ax = plt.subplots(figsize=(11, 6))
time_axis = np.arange(values.shape[1])
iqr_low = np.quantile(centered, 0.25, axis=0)
iqr_high = np.quantile(centered, 0.75, axis=0)
ax.fill_between(time_axis, iqr_low, iqr_high, color="lightgray", alpha=0.65, label="Population IQR")
ax.plot(time_axis, baseline, color="black", linewidth=2.2, label="Global baseline")
for idx, col in enumerate(rule_cols):
    rank = int(col.split("_")[1])
    member_ids = set(membership.loc[membership[col] == 1, "entity"])
    mask = np.array([country in member_ids for country in interp.index])
    if mask.sum() == 0:
        continue
    ax.plot(time_axis, centered[mask].mean(axis=0), linewidth=2.4, color=rule_colors[idx], label=f"Rule {rank} (n={mask.sum()})")
ax.set_title("DOLPHIN Rule Trajectories on World Bank GDP per Capita")
ax.set_xlabel("Aligned trajectory grid")
ax.set_ylabel("Mean-centered GDP per capita")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(OUT_DIR / "world_bank_dolphin_rule_trajectory_overlay.png", dpi=300, bbox_inches="tight")
plt.show()
"""
        ),
    ]
    return make_notebook("World Bank GDP EDA for DOLPHIN", cells)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "eda_cmie_dolphin.ipynb": cmie_notebook(),
        "eda_world_bank_gdp_dolphin.ipynb": world_bank_notebook(),
    }
    for name, nb in notebooks.items():
        path = NOTEBOOK_DIR / name
        nbf.write(nb, path)
        print(path)


if __name__ == "__main__":
    main()
