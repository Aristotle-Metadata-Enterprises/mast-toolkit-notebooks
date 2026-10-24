from dataclasses import dataclass
from itertools import combinations

import pandas as pd
from IPython.display import Markdown, display
from scipy.stats import spearmanr

from mast_toolkit_notebooks.utils import DONT_KNOW

ALPHA = 0.05
MIN_STRONG_RHO = 0.35
TOP_N = 5

# (minimum |rho|, background colour), strongest first
RHO_COLOUR_STEPS = [
    (0.65, "#90edff"),
    (0.55, "#5ffb67"),
    (0.45, "#ccf95b"),
    (0.35, "#fff2c9"),
]


@dataclass
class CorrelationMatrices:
    unadjusted_rho: pd.DataFrame
    unadjusted_p: pd.DataFrame
    adjusted_rho: pd.DataFrame
    adjusted_p: pd.DataFrame


def spearman_matrices(data):
    rho = pd.DataFrame(float("nan"), index=data.columns, columns=data.columns)
    p_values = rho.copy()
    for left, right in combinations(data.columns, 2):
        pair = data[[left, right]].dropna()  # pairwise deletion of missing values
        if len(pair) > 2 and pair[left].nunique() > 1 and pair[right].nunique() > 1:
            result = spearmanr(pair[left], pair[right])
            rho.loc[left, right] = rho.loc[right, left] = result.statistic
            p_values.loc[left, right] = p_values.loc[right, left] = result.pvalue
    return rho, p_values


def compute_correlation_matrices(unadjusted_data, adjusted_data):
    unadjusted_rho, unadjusted_p = spearman_matrices(unadjusted_data)
    adjusted_rho, adjusted_p = spearman_matrices(adjusted_data)
    return CorrelationMatrices(unadjusted_rho, unadjusted_p, adjusted_rho, adjusted_p)


def format_matrix(rho, p_values):
    cells = pd.DataFrame("n/a", index=rho.index, columns=rho.columns)
    for row in rho.index:
        for column in rho.columns:
            if row == column:
                cells.loc[row, column] = "—"
            elif pd.notna(rho.loc[row, column]):
                cells.loc[row, column] = (
                    f"{rho.loc[row, column]:.2f} (p={p_values.loc[row, column]:.3g})"
                )
    return cells


def cell_style(rho_value, p_value):
    if pd.isna(rho_value) or p_value >= ALPHA:
        return ""
    for min_rho, colour in RHO_COLOUR_STEPS:
        if abs(rho_value) > min_rho:
            return f"background-color: {colour} !important; color: #000 !important;"
    return ""


def style_matrix(rho, p_values):
    styles = rho.copy().astype(object)
    for row in rho.index:
        for column in rho.columns:
            styles.loc[row, column] = cell_style(rho.loc[row, column], p_values.loc[row, column])
    return format_matrix(rho, p_values).style.apply(lambda _: styles, axis=None)


def show_matrices(matrices, rows, columns):
    display(Markdown("#### Unadjusted: Spearman rho (p-value)"))
    display(
        style_matrix(
            matrices.unadjusted_rho.loc[rows, columns],
            matrices.unadjusted_p.loc[rows, columns],
        )
    )
    display(
        Markdown(
            "#### Adjusted: Spearman rho (p-value), "
            "excluding \"don't know\" and action rating 1"
        )
    )
    display(
        style_matrix(
            matrices.adjusted_rho.loc[rows, columns],
            matrices.adjusted_p.loc[rows, columns],
        )
    )


def strongest_candidates(survey):
    """Map each outcome variable to the variables it may be compared with."""
    candidates_by_variable = {}
    for variable in survey.beliefs_columns:
        candidates_by_variable[variable] = survey.action_columns
    for variable in [*survey.self_assess_columns, *survey.ai_outcome_columns]:
        candidates_by_variable[variable] = [*survey.action_columns, *survey.beliefs_columns]
    return candidates_by_variable


def strongest_correlations(matrices, variable, candidate_variables, question_text):
    candidates = pd.DataFrame(
        {
            "variable": candidate_variables,
            "rho": matrices.adjusted_rho.loc[candidate_variables, variable],
            "p_value": matrices.adjusted_p.loc[candidate_variables, variable],
            "question": question_text.loc[candidate_variables].to_numpy(),
        }
    )
    candidates = candidates[candidates["p_value"] < ALPHA]
    candidates = candidates.loc[candidates["rho"].abs() > MIN_STRONG_RHO]
    candidates["abs_rho"] = candidates["rho"].abs()
    candidates = candidates.sort_values("abs_rho", ascending=False)
    return candidates.head(TOP_N).drop(columns="abs_rho").reset_index(drop=True)


def display_strongest_correlations(matrices, survey):
    question_text = survey.metadata.set_index("field_name")["question text"]
    candidates_by_variable = strongest_candidates(survey)
    for variable, candidate_variables in candidates_by_variable.items():
        display(Markdown(f"#### `{variable}`: {question_text[variable]}"))
        top = strongest_correlations(matrices, variable, candidate_variables, question_text)
        display(top.style.format({"rho": "{:.2f}", "p_value": "{:.3g}"}))


def display_original_correlations(
    completed, action_analysis_columns, outcome_columns, p_threshold=0.025
):
    rows = []
    for outcome_column in outcome_columns:
        for action_column, analysis_action_column in action_analysis_columns.items():
            paired = (
                completed[[analysis_action_column, outcome_column]]
                .rename(columns={analysis_action_column: action_column})
                .replace(DONT_KNOW, float("nan"))
                .dropna()
            )
            result = spearmanr(paired[action_column], paired[outcome_column])
            rows.append(
                {
                    "outcome": outcome_column,
                    "action": action_column,
                    "spearman_rho": result.statistic,
                    "p_value": result.pvalue,
                    "n": len(paired),
                }
            )

    correlations = pd.DataFrame(rows)
    for outcome_column in outcome_columns:
        outcome_table = correlations.loc[
            (correlations["outcome"] == outcome_column)
            & (correlations["p_value"] < p_threshold),
            ["action", "spearman_rho", "p_value", "n"],
        ].sort_values("p_value")
        display(Markdown(f"### {outcome_column}"))
        display(outcome_table.reset_index(drop=True))
    return correlations
