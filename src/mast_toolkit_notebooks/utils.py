from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_NAME = "dataversity-2026"
RESPONSES_FILENAME = "dataversity_survey_2iyUvU5Z4Pdi25fBvXwQsJ_responses.csv"
METADATA_FILENAME = "dataversity_metadata.csv"

NEGATIVE = [1, 2]
NEUTRAL = [3]
POSITIVE = [4, 5]
DONT_KNOW = 9
RECODE_MAP = {
    **{value: "negative" for value in NEGATIVE},
    **{value: "neutral" for value in NEUTRAL},
    **{value: "positive" for value in POSITIVE},
    DONT_KNOW: "don't know",
}

RATING_PREFIXES = ("beliefs_", "actions_", "self_assess_", "ai_")
RECODED_PREFIXES = ("beliefs_", "self_assess_", "ai_")
DERIVED_SUFFIXES = ("_qual", "_recoded", "_excluding_1")


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


@dataclass
class SurveyData:
    responses: pd.DataFrame
    metadata: pd.DataFrame
    analysis_df: pd.DataFrame
    completed: pd.DataFrame
    rating_columns: list
    beliefs_columns: list
    action_columns: list
    action_analysis_columns: dict
    self_assess_columns: list
    ai_outcome_columns: list
    recode_columns: list


def find_data_dir(project_name=PROJECT_NAME):
    data_dir = Path("data") / project_name
    if data_dir.exists():
        return data_dir
    return Path("../data") / project_name


def load_survey(project_name=PROJECT_NAME, verbose=True):
    data_dir = find_data_dir(project_name)
    responses_file = data_dir / RESPONSES_FILENAME
    metadata_file = data_dir / METADATA_FILENAME

    responses = pd.read_csv(responses_file)
    metadata = pd.read_csv(metadata_file)
    metadata["choices_parsed"] = metadata["choices (if applicable)"].apply(parse_choices)
    if verbose:
        print(f"Loaded {len(responses):,} responses from {responses_file}")
        print(
            f"Loaded metadata for {len(metadata)} fields; "
            f"{metadata['choices_parsed'].str.len().gt(0).sum()} fields have choices."
        )
    return responses, metadata


def find_rating_columns(data):
    return [
        column
        for column in data.columns
        if column.startswith(RATING_PREFIXES) and not column.endswith(DERIVED_SUFFIXES)
    ]


def columns_with_prefix(columns, prefix):
    return [column for column in columns if column.startswith(prefix)]


def preprocess(responses):
    """Return a copy of responses with numeric ratings, excluding-1 actions and recodes."""
    analysis_df = responses.copy()
    analysis_df["response_date"] = pd.to_datetime(
        analysis_df["response_date"], errors="coerce", utc=True
    )

    rating_columns = find_rating_columns(analysis_df)
    analysis_df[rating_columns] = analysis_df[rating_columns].apply(
        pd.to_numeric, errors="coerce"
    )

    # Action rating 1 is "I don't do this", so it is masked for analysis.
    action_columns = columns_with_prefix(rating_columns, "actions_")
    for column in action_columns:
        analysis_df[f"{column}_excluding_1"] = analysis_df[column].mask(
            analysis_df[column].eq(1)
        )

    recode_columns = [
        column for column in rating_columns if column.startswith(RECODED_PREFIXES)
    ]
    for column in recode_columns:
        analysis_df[f"{column}_recoded"] = analysis_df[column].map(RECODE_MAP)
    return analysis_df


def select_completed(analysis_df):
    is_complete = analysis_df["is_complete"].astype("string").str.casefold()
    return analysis_df.loc[is_complete.eq("true")]


def prepare_survey(project_name=PROJECT_NAME, verbose=True):
    responses, metadata = load_survey(project_name, verbose=verbose)
    analysis_df = preprocess(responses)
    rating_columns = find_rating_columns(analysis_df)
    action_columns = columns_with_prefix(rating_columns, "actions_")
    recode_columns = [
        column for column in rating_columns if column.startswith(RECODED_PREFIXES)
    ]
    return SurveyData(
        responses=responses,
        metadata=metadata,
        analysis_df=analysis_df,
        completed=select_completed(analysis_df),
        rating_columns=rating_columns,
        beliefs_columns=columns_with_prefix(rating_columns, "beliefs_"),
        action_columns=action_columns,
        action_analysis_columns={column: f"{column}_excluding_1" for column in action_columns},
        self_assess_columns=columns_with_prefix(rating_columns, "self_assess_"),
        ai_outcome_columns=columns_with_prefix(rating_columns, "ai_"),
        recode_columns=recode_columns,
    )


def build_pairwise_data(survey):
    """Return (unadjusted, adjusted) completed-response ratings for pairwise analysis.

    Unadjusted keeps "don't know" (9) and action rating 1. Adjusted turns 9 into missing
    and uses the *_excluding_1 action columns.
    """
    columns = survey.rating_columns
    unadjusted_data = survey.completed[columns]

    adjusted_source = [survey.action_analysis_columns.get(column, column) for column in columns]
    adjusted_data = (
        survey.completed[adjusted_source]
        .set_axis(columns, axis=1)
        .replace(DONT_KNOW, float("nan"))
    )
    return unadjusted_data, adjusted_data
