from pathlib import Path
import sys

import pandas as pd


# ============================================================
# PROJECT ROOT / MODEL IMPORT
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

sys.path.insert(
    0,
    str(PROJECT_ROOT),
)

from src.models.condition_baseline_v3_2 import predict_condition


# ============================================================
# PATHS
# ============================================================


def get_project_paths():

    input_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "master_trail_forecast_features.csv"
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "master_forecast_condition_predictions_v3_2.csv"
    )

    today_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "master_today_condition_predictions_v3_2.csv"
    )

    return (
        input_path,
        output_path,
        today_path,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    (
        input_path,
        output_path,
        today_path,
    ) = get_project_paths()

    # ---------------------------------------------------------
    # LOAD FORECAST FEATURES
    # ---------------------------------------------------------

    print(
        "Loading master forecast features..."
    )

    dataset = pd.read_csv(
        input_path,
        parse_dates=["date"],
    )

    print(
        f"Rows loaded: "
        f"{len(dataset):,}"
    )

    print(
        f"Trails: "
        f"{dataset['trail_name'].nunique():,}"
    )

    print(
        f"Forecast dates: "
        f"{dataset['date'].nunique():,}"
    )

    print(
        f"First forecast date: "
        f"{dataset['date'].min().date()}"
    )

    print(
        f"Last forecast date: "
        f"{dataset['date'].max().date()}"
    )

    # ---------------------------------------------------------
    # VALIDATE REQUIRED MODEL INPUTS
    # ---------------------------------------------------------

    required_columns = {
        "date",
        "trail_name",
        "final_area",
        "precip_1d",
        "precip_3d",
        "precip_7d",
        "days_since_precip",
        "mean_temp_3d",
        "snowfall_3d",
        "freeze_thaw_3d",
        "minimum_elevation_feet",
        "maximum_elevation_feet",
        "north_facing_pct",
        "south_facing_pct",
    }

    missing_columns = (
        required_columns
        - set(dataset.columns)
    )

    if missing_columns:

        raise RuntimeError(
            "Forecast feature dataset is missing "
            "required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    duplicate_input_rows = (
        dataset
        .duplicated(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sum()
    )

    if duplicate_input_rows > 0:

        raise RuntimeError(
            f"Found {duplicate_input_rows} duplicate "
            "trail/date rows before prediction."
        )

    # ---------------------------------------------------------
    # RUN V3.2
    # ---------------------------------------------------------

    print()
    print(
        "Running v3.2 hero-dirt model "
        "across 7-day forecast..."
    )

    predictions = dataset.apply(
        predict_condition,
        axis=1,
    )

    results = pd.concat(
        [
            dataset.reset_index(
                drop=True
            ),
            predictions.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # DAILY RANK
    # ---------------------------------------------------------

    results[
        "daily_rank"
    ] = (
        results
        .groupby("date")[
            "rideability_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # ---------------------------------------------------------
    # AREA RANK
    # ---------------------------------------------------------

    results[
        "area_rank"
    ] = (
        results
        .groupby(
            [
                "date",
                "final_area",
            ]
        )[
            "rideability_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    # ---------------------------------------------------------
    # SORT
    # ---------------------------------------------------------

    results = (
        results
        .sort_values(
            [
                "date",
                "rideability_score",
                "trail_name",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    duplicate_rows = (
        results
        .duplicated(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sum()
    )

    missing_predictions = (
        results[
            [
                "estimated_moisture",
                "surface_state",
                "rideability_score",
                "rideability",
                "reason",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    invalid_scores = (
        (
            results[
                "rideability_score"
            ]
            < 0
        )
        |
        (
            results[
                "rideability_score"
            ]
            > 100
        )
    ).sum()

    unique_trails = (
        results[
            "trail_name"
        ]
        .nunique()
    )

    unique_dates = (
        results[
            "date"
        ]
        .nunique()
    )

    expected_rows = (
        unique_trails
        * unique_dates
    )

    # ---------------------------------------------------------
    # SAVE FULL FORECAST
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SAVE FIRST / CURRENT FORECAST DATE
    # ---------------------------------------------------------

    first_date = (
        results[
            "date"
        ]
        .min()
    )

    today = (
        results[
            results[
                "date"
            ]
            == first_date
        ]
        .copy()
        .sort_values(
            [
                "rideability_score",
                "trail_name",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    today.to_csv(
        today_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MASTER V3.2 FORECAST PREDICTIONS COMPLETE"
    )
    print("=" * 72)

    print(
        f"Prediction rows: "
        f"{len(results):,}"
    )

    print(
        f"Expected rows: "
        f"{expected_rows:,}"
    )

    print(
        f"Trails: "
        f"{unique_trails:,}"
    )

    print(
        f"Forecast dates: "
        f"{unique_dates:,}"
    )

    print(
        f"Duplicate trail/date rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"Missing predictions: "
        f"{missing_predictions:,}"
    )

    print(
        f"Scores outside 0-100: "
        f"{invalid_scores:,}"
    )

    print()
    print(
        f"Full 7-day predictions:"
        f"\n  {output_path}"
    )

    print()
    print(
        f"First-date predictions:"
        f"\n  {today_path}"
    )

    # ---------------------------------------------------------
    # FIRST-DATE DISTRIBUTIONS
    # ---------------------------------------------------------

    print()
    print(
        f"First forecast date: "
        f"{first_date.date()}"
    )

    print()
    print(
        "Surface-state distribution:"
    )

    print()

    print(
        today[
            "surface_state"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Rideability distribution:"
    )

    print()

    print(
        today[
            "rideability"
        ]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # TOP 30
    # ---------------------------------------------------------

    print()
    print(
        "Top 30 trails today:"
    )

    print()

    top_columns = [
        "daily_rank",
        "trail_name",
        "final_area",
        "rideability_score",
        "rideability",
        "surface_state",
        "estimated_moisture",
        "precip_1d",
        "precip_3d",
        "precip_7d",
        "days_since_precip",
        "mean_elevation_feet",
        "reason",
    ]

    top_columns = [
        column
        for column in top_columns
        if column in today.columns
    ]

    print(
        today[
            top_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # BOTTOM 20
    # ---------------------------------------------------------

    print()
    print(
        "Bottom 20 trails today:"
    )

    print()

    print(
        today[
            top_columns
        ]
        .tail(20)
        .sort_values(
            [
                "rideability_score",
                "trail_name",
            ],
            ascending=[
                True,
                True,
            ],
        )
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # AREA SUMMARY
    # ---------------------------------------------------------

    print()
    print(
        "Area summary today:"
    )

    print()

    area_summary = (
        today
        .groupby(
            "final_area",
            as_index=False,
        )
        .agg(
            trail_count=(
                "trail_name",
                "size",
            ),
            mean_score=(
                "rideability_score",
                "mean",
            ),
            median_score=(
                "rideability_score",
                "median",
            ),
            best_score=(
                "rideability_score",
                "max",
            ),
            worst_score=(
                "rideability_score",
                "min",
            ),
        )
        .sort_values(
            "mean_score",
            ascending=False,
        )
    )

    print(
        area_summary
        .round(
            {
                "mean_score": 1,
                "median_score": 1,
                "best_score": 1,
                "worst_score": 1,
            }
        )
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # DAILY REGIONAL OUTLOOK
    # ---------------------------------------------------------

    print()
    print(
        "7-day regional outlook:"
    )

    print()

    daily_summary = (
        results
        .groupby(
            "date",
            as_index=False,
        )
        .agg(
            mean_score=(
                "rideability_score",
                "mean",
            ),
            median_score=(
                "rideability_score",
                "median",
            ),
            excellent_trails=(
                "rideability",
                lambda s:
                    (s == "EXCELLENT").sum(),
            ),
            good_trails=(
                "rideability",
                lambda s:
                    (s == "GOOD").sum(),
            ),
            fair_trails=(
                "rideability",
                lambda s:
                    (s == "FAIR").sum(),
            ),
            poor_trails=(
                "rideability",
                lambda s:
                    (s == "POOR").sum(),
            ),
            avoid_trails=(
                "rideability",
                lambda s:
                    (s == "AVOID").sum(),
            ),
        )
    )

    daily_summary[
        "date"
    ] = (
        daily_summary[
            "date"
        ]
        .dt.date
    )

    print(
        daily_summary
        .round(
            {
                "mean_score": 1,
                "median_score": 1,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()