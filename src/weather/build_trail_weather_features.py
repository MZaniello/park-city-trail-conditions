from pathlib import Path

import numpy as np
import pandas as pd


def calculate_days_since_precip(series):
    """
    Calculate the number of consecutive days since measurable
    precipitation for one trail.
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
    # ---------------------------------------------------------
    # FILE PATHS
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    weather_path = (
        project_root
        / "data"
        / "raw"
        / "trail_historical_weather.csv"
    )

    terrain_path = (
        project_root
        / "data"
        / "processed"
        / "trail_terrain_summary.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "trail_modeling_dataset.csv"
    )

    # ---------------------------------------------------------
    # LOAD WEATHER
    # ---------------------------------------------------------

    print("Loading trail-specific weather...")

    weather = pd.read_csv(
        weather_path,
        parse_dates=["date"],
    )

    print(f"Weather rows: {len(weather):,}")
    print(
        f"Trails in weather data: "
        f"{weather['trail_name'].nunique()}"
    )

    # ---------------------------------------------------------
    # SORT BY TRAIL AND DATE
    #
    # This is important. Rolling windows must be calculated
    # chronologically and separately for each trail.
    # ---------------------------------------------------------

    weather = weather.sort_values(
        ["trail_name", "date"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # PRECIPITATION FEATURES
    # ---------------------------------------------------------

    print("Building precipitation features...")

    weather["precip_1d"] = weather["precipitation_sum"]

    weather["precip_3d"] = (
        weather
        .groupby("trail_name")["precipitation_sum"]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=1,
            ).sum()
        )
    )

    weather["precip_7d"] = (
        weather
        .groupby("trail_name")["precipitation_sum"]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).sum()
        )
    )

    # ---------------------------------------------------------
    # TEMPERATURE FEATURES
    # ---------------------------------------------------------

    print("Building temperature features...")

    weather["mean_temp_3d"] = (
        weather
        .groupby("trail_name")["temperature_2m_mean"]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=1,
            ).mean()
        )
    )

    weather["mean_temp_7d"] = (
        weather
        .groupby("trail_name")["temperature_2m_mean"]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    # ---------------------------------------------------------
    # FREEZE / THAW FEATURES
    #
    # A freeze-thaw day is one where the minimum temperature
    # falls below 32°F and the maximum rises above 32°F.
    # ---------------------------------------------------------

    print("Building freeze-thaw features...")

    weather["freeze_thaw"] = (
        (weather["temperature_2m_min"] < 32)
        & (weather["temperature_2m_max"] > 32)
    ).astype(int)

    weather["freeze_thaw_3d"] = (
        weather
        .groupby("trail_name")["freeze_thaw"]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=1,
            ).sum()
        )
    )

    weather["freeze_thaw_7d"] = (
        weather
        .groupby("trail_name")["freeze_thaw"]
        .transform(
            lambda x: x.rolling(
                window=7,
                min_periods=1,
            ).sum()
        )
    )

    # ---------------------------------------------------------
    # SNOWFALL FEATURES
    # ---------------------------------------------------------

    print("Building snowfall features...")

    weather["snowfall_3d"] = (
        weather
        .groupby("trail_name")["snowfall_sum"]
        .transform(
            lambda x: x.rolling(
                window=3,
                min_periods=1,
            ).sum()
        )
    )

    weather["snowfall_7d"] = (
        weather
        .groupby("trail_name")["snowfall_sum"]
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

    print("Calculating days since precipitation...")

    weather["days_since_precip"] = (
        weather
        .groupby("trail_name")["precipitation_sum"]
        .transform(calculate_days_since_precip)
    )

    # ---------------------------------------------------------
    # LOAD TERRAIN DATA
    # ---------------------------------------------------------

    print("Loading terrain features...")

    terrain = pd.read_csv(terrain_path)

    print(
        f"Trails in terrain data: "
        f"{terrain['trail_name'].nunique()}"
    )

    # ---------------------------------------------------------
    # JOIN WEATHER + TERRAIN
    # ---------------------------------------------------------

    print("Joining weather and terrain...")

    dataset = weather.merge(
        terrain,
        on="trail_name",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # SORT FINAL DATASET
    # ---------------------------------------------------------

    dataset = dataset.sort_values(
        ["date", "trail_name"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    print("Validating modeling dataset...")

    missing_terrain = (
        dataset["minimum_elevation_feet"]
        .isna()
        .sum()
    )

    duplicate_rows = dataset.duplicated(
        subset=["date", "trail_name"]
    ).sum()

    trail_count = dataset[
        "trail_name"
    ].nunique()

    date_count = dataset[
        "date"
    ].nunique()

    # Each trail/date combination should appear exactly once.
    expected_rows = trail_count * date_count

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print("\nModeling dataset created!")
    print(f"Rows: {len(dataset):,}")
    print(f"Expected rows: {expected_rows:,}")
    print(f"Trails: {trail_count}")
    print(f"Dates: {date_count:,}")
    print(f"Columns: {len(dataset.columns)}")
    print(f"Duplicate trail/date rows: {duplicate_rows:,}")
    print(f"Missing terrain rows: {missing_terrain:,}")

    print(f"\nSaved to: {output_path}")

    # ---------------------------------------------------------
    # SHOW ONE DAY AS A SANITY CHECK
    # ---------------------------------------------------------

    example_date = pd.Timestamp("2026-08-01")

    example = dataset[
        dataset["date"] == example_date
    ]

    print("\nExample — August 1, 2026:")

    print(
        example[
            [
                "trail_name",
                "temperature_2m_mean",
                "precip_1d",
                "precip_3d",
                "precip_7d",
                "days_since_precip",
                "minimum_elevation_feet",
                "maximum_elevation_feet",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()