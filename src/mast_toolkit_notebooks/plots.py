import matplotlib.pyplot as plt
import pandas as pd

PLOT_WIDTH = 18

recoded_choice_labels = {
    "negative": "Negative (Strongly Disagree, Disagree)",
    "neutral": "Neutral (Neither Agree nor Disagree)",
    "positive": "Positive (Agree, Strongly Agree)",
}


def choice_labels(metadata_by_field, field):
    return {
        int(choice["code"]): choice["value"]
        for choice in metadata_by_field.loc[field, "choices_parsed"]
    }


def to_percent_exc_nan(counts):
    row_totals = counts.sum(axis=1).replace(0, float("nan"))
    return counts.div(row_totals, axis=0) * 100


def plot_stacked_percent_barh(percentages, colors, labels, title):
    """Plot rows of percentages (columns are stacked in order) as horizontal bars."""
    figure, axis = plt.subplots(figsize=(PLOT_WIDTH, 12))
    figure.subplots_adjust(left=0.38, right=0.96, top=0.88, bottom=0.25)
    left = [0.0] * len(percentages)
    positions = range(len(percentages))
    for code in percentages.columns:
        widths = percentages[code].fillna(0).to_list()
        axis.barh(
            positions,
            widths,
            left=left,
            height=0.75,
            color=colors[code],
            edgecolor="white",
            label=labels[code],
        )
        for position, width, start in zip(positions, widths, left):
            if width >= 5:
                axis.text(
                    start + width / 2,
                    position,
                    f"{width:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white" if code in (1, 5, 9) else "black",
                )
        left = [start + width for start, width in zip(left, widths)]

    axis.set_yticks(list(positions), labels=percentages.index, fontsize=18)
    axis.tick_params(axis="x", labelsize=16)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Responses (%)", fontsize=18)
    axis.set_title(title, fontsize=24, pad=0)
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    figure.legend(
        *axis.get_legend_handles_labels(),
        fontsize=16,
        ncol=len(percentages.columns),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    plt.show()


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
    kept_categories = category_counts.index[category_counts >= 5]
    # Keep the largest categories so at most max_labels remain, including "Other".
    max_labels = 10
    if len(kept_categories) < len(category_counts) or len(kept_categories) > max_labels:
        kept_categories = kept_categories[: max_labels - 1]
    chart_data["category_group"] = chart_data["category_label"].where(
        chart_data["category_label"].isin(kept_categories), "Other"
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
            # The mean of an empty selection is NaN, so empty categories stay blank.
            positive_percentages.loc[category, metric] = valid_scores.isin([4, 5]).mean() * 100

    figure, axis = plt.subplots(figsize=(PLOT_WIDTH, 12))
    figure.subplots_adjust(left=0.38, right=0.96, top=0.88, bottom=0.25)
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
        axis.barh(
            [position + offset for position in positions],
            positive_percentages[metric].to_numpy(),
            height=bar_width,
            color=colors[metric_index % len(colors)],
            label=metric_label,
        )

    axis.set_yticks(
        list(positions),
        labels=[f"{category} (n={group_counts[category]})" for category in categories],
        fontsize=16,
    )
    axis.tick_params(axis="x", labelsize=16)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("Percent of positive responses (ratings 'agree' or 'strongly agree')", fontsize=16)
    axis.set_ylabel(f"{category_column.replace('_', ' ').title()} (n = completed responses)", fontsize=16)
    axis.set_title("Percent of positive responses: " + title, fontsize=20)
    axis.grid(axis="x", alpha=0.2)
    axis.set_axisbelow(True)
    figure.legend(
        *axis.get_legend_handles_labels(),
        fontsize=16,
        ncol=len(metric_columns),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
    )
    plt.show()
