from pathlib import Path

import numpy as np
import pandas as pd


def calculate_days_since_precip(series):
    """
    Calculate consecutive days since measurable precipitation.
    """

    counter = np.nan
    values = []

    for precipitation in series:

        if precipitation >= 0.01:
            counter = 0

        elif not np.isnan(counter):
            counter += 1

        values.append(counter)

    return pd.Series(
        values,
        index=series.index,
    )


def main():

    project_root = Path(__file__).resolve().parents[2]

    historical_path = (
        project_root
        / "data"
        / "raw"
        / "trail_historical_weather.csv"
    )

    forecast_path = (
        project_root
        / "data"
        / "raw"
        / "trail_forecast.csv"
    )

    terrain_path = (
        project_root
        / "data"
        / "processed"
        / "trail_terrain_summary.csv"
    )

    topography_path = (
        project_root
        / "data"
        / "processed"
        / "trail_topography_features.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "trail_forecast_features.csv"
    )

    # ---------------------------------------------------------
    # LOAD WEATHER
    # ---------------------------------------------------------

    print("Loading historical weather...")

    historical = pd.read_csv(
        historical_path,
        parse_dates=["date"],
    )

    print("Loading forecast weather...")

    forecast = pd.read_csv(
        forecast_path,
        parse_dates=["date"],
    )

    forecast_dates = set(
        forecast["date"].unique()
    )

    # ---------------------------------------------------------
    # REMOVE OVERLAPPING HISTORICAL DATES
    #
    # Today's date can exist in both files.
    # For forecast prediction, use the forecast version.
    # ---------------------------------------------------------

    historical = historical[
        ~historical["date"].isin(
            forecast_dates
        )
    ].copy()

    # ---------------------------------------------------------
    # IDENTIFY SOURCE
    # ---------------------------------------------------------

    historical["weather_source"] = "historical"
    forecast["weather_source"] = "forecast"

    # ---------------------------------------------------------
    # COMBINE
    # ---------------------------------------------------------

    print(
        "Combining recent history with forecast..."
    )

    weather = pd.concat(
        [
            historical,
            forecast,
        ],
        ignore_index=True,
    )

    weather = weather.sort_values(
        [
            "trail_name",
            "date",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # PRECIPITATION
    # ---------------------------------------------------------

    print("Building precipitation features...")

    weather["precip_1d"] = (
        weather["precipitation_sum"]
    )

    weather["precip_3d"] = (
        weather
        .groupby("trail_name")[
            "precipitation_sum"
        ]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=1,
            ).sum()
        )
    )

    weather["precip_7d"] = (
        weather
        .groupby("trail_name")[
            "precipitation_sum"
        ]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).sum()
        )
    )

    # ---------------------------------------------------------
    # TEMPERATURE
    # ---------------------------------------------------------

    print("Building temperature features...")

    weather["mean_temp_3d"] = (
        weather
        .groupby("trail_name")[
            "temperature_2m_mean"
        ]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=1,
            ).mean()
        )
    )

    weather["mean_temp_7d"] = (
        weather
        .groupby("trail_name")[
            "temperature_2m_mean"
        ]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    # ---------------------------------------------------------
    # FREEZE / THAW
    # ---------------------------------------------------------

    print("Building freeze-thaw features...")

    weather["freeze_thaw"] = (
        (weather["temperature_2m_min"] < 32)
        & (weather["temperature_2m_max"] > 32)
    ).astype(int)

    weather["freeze_thaw_3d"] = (
        weather
        .groupby("trail_name")[
            "freeze_thaw"
        ]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=1,
            ).sum()
        )
    )

    weather["freeze_thaw_7d"] = (
        weather
        .groupby("trail_name")[
            "freeze_thaw"
        ]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).sum()
        )
    )

    # ---------------------------------------------------------
    # SNOW
    # ---------------------------------------------------------

    print("Building snowfall features...")

    weather["snowfall_3d"] = (
        weather
        .groupby("trail_name")[
            "snowfall_sum"
        ]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=1,
            ).sum()
        )
    )

    weather["snowfall_7d"] = (
        weather
        .groupby("trail_name")[
            "snowfall_sum"
        ]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).sum()
        )
    )

    # ---------------------------------------------------------
    # DAYS SINCE PRECIPITATION
    # ---------------------------------------------------------

    print(
        "Calculating days since precipitation..."
    )

    weather["days_since_precip"] = (
        weather
        .groupby("trail_name")[
            "precipitation_sum"
        ]
        .transform(
            calculate_days_since_precip
        )
    )

    # ---------------------------------------------------------
    # KEEP FORECAST ROWS ONLY
    # ---------------------------------------------------------

    forecast_features = weather[
        weather["weather_source"]
        == "forecast"
    ].copy()

    # ---------------------------------------------------------
    # TERRAIN FEATURES
    # ---------------------------------------------------------

    print("Loading terrain features...")

    terrain = pd.read_csv(
        terrain_path
    )

    topography = pd.read_csv(
        topography_path
    )

    forecast_features = (
        forecast_features.merge(
            terrain,
            on="trail_name",
            how="left",
            validate="many_to_one",
        )
    )

    topography_columns = [
        "trail_name",
        "mean_slope_degrees",
        "median_slope_degrees",
        "north_facing_pct",
        "east_facing_pct",
        "south_facing_pct",
        "west_facing_pct",
        "topography_sample_count",
    ]

    forecast_features = (
        forecast_features.merge(
            topography[
                topography_columns
            ],
            on="trail_name",
            how="left",
            validate="many_to_one",
        )
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    forecast_features = (
        forecast_features.sort_values(
            [
                "date",
                "trail_name",
            ]
        ).reset_index(drop=True)
    )

    duplicate_rows = (
        forecast_features.duplicated(
            subset=[
                "date",
                "trail_name",
            ]
        ).sum()
    )

    missing_terrain = (
        forecast_features[
            "minimum_elevation_feet"
        ]
        .isna()
        .sum()
    )

    missing_topography = (
        forecast_features[
            "mean_slope_degrees"
        ]
        .isna()
        .sum()
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_features.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("FORECAST FEATURES COMPLETE")
    print("=" * 60)

    print(
        f"Rows: "
        f"{len(forecast_features):,}"
    )

    print(
        f"Trails: "
        f"{forecast_features['trail_name'].nunique()}"
    )

    print(
        f"Forecast dates: "
        f"{forecast_features['date'].nunique()}"
    )

    print(
        f"Duplicate trail/date rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"Missing terrain rows: "
        f"{missing_terrain:,}"
    )

    print(
        f"Missing topography rows: "
        f"{missing_topography:,}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    # ---------------------------------------------------------
    # FIRST DATE PREVIEW
    # ---------------------------------------------------------

    first_date = (
        forecast_features["date"].min()
    )

    print()
    print(
        f"Preview — {first_date.date()}:"
    )

    print(
        forecast_features[
            forecast_features["date"]
            == first_date
        ][
            [
                "trail_name",
                "temperature_2m_mean",
                "precip_1d",
                "precip_3d",
                "precip_7d",
                "days_since_precip",
                "minimum_elevation_feet",
                "south_facing_pct",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
    