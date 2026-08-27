from pathlib import Path
import sys

import pandas as pd


# ============================================================
# IMPORT EXISTING V3.2 MODEL
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
# MAIN
# ============================================================


def main():

    input_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "master_trail_modeling_dataset.csv"
    )

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "master_condition_predictions_v3_2.csv"
    )

    latest_output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "master_latest_condition_predictions_v3_2.csv"
    )

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    print(
        "Loading master modeling dataset..."
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
        f"Dates: "
        f"{dataset['date'].nunique():,}"
    )

    # ---------------------------------------------------------
    # VALIDATE REQUIRED MODEL COLUMNS
    # ---------------------------------------------------------

    required_columns = {
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
            "Modeling dataset is missing required "
            "columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    # ---------------------------------------------------------
    # GENERATE PREDICTIONS
    # ---------------------------------------------------------

    print()
    print(
        "Running v3.2 hero-dirt model "
        "across all trail-date rows..."
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
    # DAILY RANKING
    # ---------------------------------------------------------

    print(
        "Calculating daily trail rankings..."
    )

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
    # VALIDATE
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

    # ---------------------------------------------------------
    # SAVE FULL HISTORY
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
    # SAVE LATEST DATE
    # ---------------------------------------------------------

    latest_date = (
        results["date"]
        .max()
    )

    latest = (
        results[
            results[
                "date"
            ]
            == latest_date
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

    latest[
        "daily_rank"
    ] = (
        latest.index
        + 1
    )

    latest.to_csv(
        latest_output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MASTER V3.2 CONDITION MODEL COMPLETE"
    )
    print("=" * 72)

    print(
        f"Prediction rows: "
        f"{len(results):,}"
    )

    print(
        f"Trails: "
        f"{results['trail_name'].nunique():,}"
    )

    print(
        f"Dates: "
        f"{results['date'].nunique():,}"
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
        f"Full historical predictions:"
        f"\n  {output_path}"
    )

    print()
    print(
        f"Latest-date predictions:"
        f"\n  {latest_output_path}"
    )

    # ---------------------------------------------------------
    # LATEST-DATE CONDITION DISTRIBUTION
    # ---------------------------------------------------------

    print()
    print(
        f"Latest prediction date: "
        f"{latest_date.date()}"
    )

    print()
    print(
        "Surface-state distribution:"
    )

    print()

    print(
        latest[
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
        latest[
            "rideability"
        ]
        .value_counts()
        .to_string()
    )

    # ---------------------------------------------------------
    # BEST TRAILS
    # ---------------------------------------------------------

    print()
    print(
        "Top 30 trails:"
    )

    print()

    print(
        latest[
            [
                "daily_rank",
                "trail_name",
                "final_area",
                "rideability_score",
                "rideability",
                "surface_state",
                "estimated_moisture",
                "precip_1d",
                "precip_3d",
                "days_since_precip",
                "mean_elevation_feet",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # WORST TRAILS
    # ---------------------------------------------------------

    print()
    print(
        "Bottom 20 trails:"
    )

    print()

    print(
        latest[
            [
                "daily_rank",
                "trail_name",
                "final_area",
                "rideability_score",
                "rideability",
                "surface_state",
                "estimated_moisture",
                "precip_1d",
                "precip_3d",
                "days_since_precip",
                "mean_elevation_feet",
            ]
        ]
        .tail(20)
        .sort_values(
            "rideability_score"
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()