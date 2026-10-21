import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Markdown, display
from scipy.stats import spearmanr


def parse_choices(choices):
    if pd.isna(choices) or not choices:
        return []

    parsed = []
    for choice in choices.split("|"):
        code, separator, value = choice.partition(";")
        if not separator:
            raise ValueError(f"Malformed choice entry: {choice!r}")
        parsed.append({"code": code, "value": value})
    return parsed


def display_original_correlations(
    completed, action_analysis_columns, outcome_columns, p_threshold=0.025
):
    rows = []
    for outcome_column in outcome_columns:
        for action_column, analysis_action_column in action_analysis_columns.items():
            paired = (
                completed[[analysis_action_column, outcome_column]]
                .rename(columns={analysis_action_column: action_column})
                .replace(9, float("nan"))
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


def plot_positive_metrics_by_category(data, category_column, metric_columns, title):
    chart_data = data[[category_column, *metric_columns]].copy()
    chart_data["category_label"] = (
        chart_data[category_column]
        .fillna("Unspecified")
        .astype("string")
        .str.strip()
        .replace("", "Unspecified")
    )

    category_counts = chart_data["category_label"].value_counts()
    small_categories = category_counts.index[category_counts < 5]
    chart_data["category_group"] = chart_data["category_label"].where(
        ~chart_data["category_label"].isin(small_categories), "Other"
    )
    group_counts = chart_data["category_group"].value_counts()
    categories = sorted(
        [category for category in group_counts.index if category != "Other"],
        key=str.casefold,
    )
    if "Other" in group_counts.index:
        categories.append("Other")

    positive_percentages = pd.DataFrame(
        index=categories, columns=metric_columns, dtype=float
    )
    for category in categories:
        category_responses = chart_data.loc[chart_data["category_group"] == category]
        for metric in metric_columns:
            valid_scores = category_responses[metric].loc[
                category_responses[metric].isin([1, 2, 3, 4, 5, 9])
            ]
            positive_percentages.loc[category, metric] = (
                valid_scores.isin([4, 5]).mean() * 100
                if not valid_scores.empty
                else float("nan")
            )

    figure_width = max(14, len(categories) * 0.55)
    figure, axis = plt.subplots(figsize=(figure_width, 7))
    positions = range(len(categories))
    bar_width = min(0.24, 0.8 / len(metric_columns))
    colors = ["#397b69", "#d28b45", "#547da5"]

    for metric_index, metric in enumerate(metric_columns):
        offset = (metric_index - (len(metric_columns) - 1) / 2) * bar_width
        metric_label = (
            metric.removeprefix("ai_")
            .removeprefix("self_assess_")
            .replace("_", " ")
            .title()
        )
        axis.bar(
            [position + offset for position in positions],
            positive_percentages[metric].to_numpy(),
            width=bar_width,
            color=colors[metric_index % len(colors)],
            label=metric_label,
        )

    axis.set_xticks(
        list(positions),
        labels=[f"{category}\n(n={group_counts[category]})" for category in categories],
        rotation=45,
        ha="right",
    )
    axis.set_ylim(0, 100)
    axis.set_ylabel("Positive responses (ratings 4–5, % of valid metric responses)")
    axis.set_xlabel(f"{category_column.replace('_', ' ').title()} (n = completed responses)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    axis.legend(title="Outcome")
    figure.tight_layout()
    plt.show()
