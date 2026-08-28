from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_trail_forecast_features.csv"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "master_forecast_condition_predictions_v3_2.csv"
)


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column
    return None


def main():

    print("=" * 72)
    print("CURRENT FORECAST / MODEL VALIDATION")
    print("=" * 72)

    features = pd.read_csv(FEATURES_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    features["date"] = pd.to_datetime(features["date"])
    predictions["date"] = pd.to_datetime(predictions["date"])

    print(f"\nFeature rows: {len(features):,}")
    print(f"Prediction rows: {len(predictions):,}")
    print(f"Feature trails: {features['trail_name'].nunique():,}")
    print(f"Prediction trails: {predictions['trail_name'].nunique():,}")

    # --------------------------------------------------------
    # Detect useful columns
    # --------------------------------------------------------

    precip_1d = find_column(
        features,
        ["precip_1d", "precipitation_sum"],
    )

    precip_3d = find_column(
        features,
        ["precip_3d"],
    )

    precip_7d = find_column(
        features,
        ["precip_7d"],
    )

    temp = find_column(
        features,
        [
            "temperature_2m_mean",
            "temperature_mean",
            "temp_mean",
        ],
    )

    weather_code = find_column(
        features,
        ["weather_code"],
    )

    score = find_column(
        predictions,
        [
            "rideability_score",
            "condition_score",
            "score",
        ],
    )

    surface = find_column(
        predictions,
        [
            "surface_state",
            "predicted_surface_state",
        ],
    )

    rideability = find_column(
        predictions,
        [
            "rideability",
            "rideability_label",
            "condition",
        ],
    )

    print("\nDetected columns:")
    print(f"  1-day precipitation: {precip_1d}")
    print(f"  3-day precipitation: {precip_3d}")
    print(f"  7-day precipitation: {precip_7d}")
    print(f"  Mean temperature:     {temp}")
    print(f"  Weather code:         {weather_code}")
    print(f"  Score:                {score}")
    print(f"  Surface state:        {surface}")
    print(f"  Rideability label:    {rideability}")

    if score is None:
        raise RuntimeError(
            "Could not find rideability score column."
        )

    # --------------------------------------------------------
    # Merge features + predictions
    # --------------------------------------------------------

    feature_columns = ["trail_name", "date"]

    for column in [
        precip_1d,
        precip_3d,
        precip_7d,
        temp,
        weather_code,
    ]:
        if column is not None and column not in feature_columns:
            feature_columns.append(column)

    prediction_columns = [
        "trail_name",
        "date",
        score,
    ]

    for column in [surface, rideability]:
        if column is not None and column not in prediction_columns:
            prediction_columns.append(column)

    merged = predictions[prediction_columns].merge(
        features[feature_columns],
        on=["trail_name", "date"],
        how="left",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Basic integrity
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("INTEGRITY")
    print("=" * 72)

    duplicates = merged.duplicated(
        subset=["trail_name", "date"]
    ).sum()

    missing_scores = merged[score].isna().sum()

    print(f"Duplicate trail/date rows: {duplicates:,}")
    print(f"Missing scores: {missing_scores:,}")
    print(
        f"Score range: "
        f"{merged[score].min():.1f} - "
        f"{merged[score].max():.1f}"
    )

    # --------------------------------------------------------
    # Regional daily summary
    # --------------------------------------------------------

    aggregations = {
        score: ["mean", "median", "min", "max"],
    }

    if precip_1d:
        aggregations[precip_1d] = ["mean", "max"]

    if precip_3d:
        aggregations[precip_3d] = ["mean", "max"]

    if precip_7d:
        aggregations[precip_7d] = ["mean", "max"]

    if temp:
        aggregations[temp] = ["mean"]

    daily = (
        merged
        .groupby("date")
        .agg(aggregations)
        .round(2)
    )

    daily.columns = [
        "_".join(column).strip()
        for column in daily.columns
    ]

    print("\n" + "=" * 72)
    print("REGIONAL DAILY OUTLOOK")
    print("=" * 72)

    print(daily.to_string())

    # --------------------------------------------------------
    # Surface-state distribution
    # --------------------------------------------------------

    if surface:

        print("\n" + "=" * 72)
        print("SURFACE STATES BY DATE")
        print("=" * 72)

        surface_table = pd.crosstab(
            merged["date"],
            merged[surface],
        )

        print(surface_table.to_string())

    # --------------------------------------------------------
    # Rideability distribution
    # --------------------------------------------------------

    if rideability:

        print("\n" + "=" * 72)
        print("RIDEABILITY BY DATE")
        print("=" * 72)

        rideability_table = pd.crosstab(
            merged["date"],
            merged[rideability],
        )

        print(rideability_table.to_string())

    # --------------------------------------------------------
    # Lowest scoring trail-days
    # --------------------------------------------------------

    display_columns = [
        "date",
        "trail_name",
        score,
    ]

    for column in [
        surface,
        rideability,
        precip_1d,
        precip_3d,
        precip_7d,
        temp,
        weather_code,
    ]:
        if column is not None and column not in display_columns:
            display_columns.append(column)

    worst = (
        merged
        .sort_values(score)
        [display_columns]
        .head(25)
    )

    print("\n" + "=" * 72)
    print("25 LOWEST-SCORING TRAIL DAYS")
    print("=" * 72)

    print(
        worst.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Biggest day-to-day regional changes
    # --------------------------------------------------------

    regional_scores = (
        merged
        .groupby("date")[score]
        .mean()
        .sort_index()
    )

    score_change = regional_scores.diff()

    changes = pd.DataFrame(
        {
            "mean_score": regional_scores,
            "change_from_previous_day": score_change,
        }
    ).round(2)

    print("\n" + "=" * 72)
    print("DAY-TO-DAY REGIONAL SCORE CHANGES")
    print("=" * 72)

    print(changes.to_string())

    print("\n" + "=" * 72)
    print("VALIDATION SCRIPT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()