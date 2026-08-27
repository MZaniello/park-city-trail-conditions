from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

PRECIP_THRESHOLD_INCHES = 0.01


# ============================================================
# PATHS
# ============================================================


def get_project_paths():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    weather_path = (
        project_root
        / "data"
        / "raw"
        / "master_trail_historical_weather.csv"
    )

    topography_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_topography_features.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_modeling_dataset.csv"
    )

    return (
        weather_path,
        topography_path,
        output_path,
    )


# ============================================================
# DAYS SINCE PRECIPITATION
# ============================================================


def calculate_days_since_precip(
    precipitation,
):

    """
    Calculate number of consecutive dry days.

    A day with at least PRECIP_THRESHOLD_INCHES
    counts as a precipitation day.

    Example:

        precipitation:
        0.10, 0.00, 0.00, 0.03

        days_since_precip:
        0, 1, 2, 0
    """

    values = (
        precipitation
        .fillna(0)
        .to_numpy()
    )

    result = np.zeros(
        len(values),
        dtype=int,
    )

    dry_days = 0

    for index, value in enumerate(
        values
    ):

        if (
            value
            >= PRECIP_THRESHOLD_INCHES
        ):

            dry_days = 0

        else:

            dry_days += 1

        result[index] = dry_days

    return pd.Series(
        result,
        index=precipitation.index,
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================


def build_weather_features(
    trail_weather,
):

    """
    Build rolling weather features separately
    for every trail.
    """

    trail_weather = (
        trail_weather
        .sort_values(
            [
                "trail_name",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # PRECIPITATION
    # ---------------------------------------------------------

    trail_weather[
        "precip_1d"
    ] = trail_weather[
        "precipitation_sum"
    ]

    trail_weather[
        "precip_3d"
    ] = (
        trail_weather
        .groupby(
            "trail_name"
        )[
            "precipitation_sum"
        ]
        .transform(
            lambda series:
                series.rolling(
                    window=3,
                    min_periods=1,
                ).sum()
        )
    )

    trail_weather[
        "precip_7d"
    ] = (
        trail_weather
        .groupby(
            "trail_name"
        )[
            "precipitation_sum"
        ]
        .transform(
            lambda series:
                series.rolling(
                    window=7,
                    min_periods=1,
                ).sum()
        )
    )

    # ---------------------------------------------------------
    # TEMPERATURE
    # ---------------------------------------------------------

    trail_weather[
        "mean_temp_3d"
    ] = (
        trail_weather
        .groupby(
            "trail_name"
        )[
            "temperature_2m_mean"
        ]
        .transform(
            lambda series:
                series.rolling(
                    window=3,
                    min_periods=1,
                ).mean()
        )
    )

    trail_weather[
        "mean_temp_7d"
    ] = (
        trail_weather
        .groupby(
            "trail_name"
        )[
            "temperature_2m_mean"
        ]
        .transform(
            lambda series:
                series.rolling(
                    window=7,
                    min_periods=1,
                ).mean()
        )
    )

    # ---------------------------------------------------------
    # FREEZE / THAW
    # ---------------------------------------------------------
    #
    # A freeze-thaw day is defined as:
    #
    # minimum temperature <= 32°F
    # AND
    # maximum temperature > 32°F
    #
    # Then count how many occurred in the previous
    # three/seven days.
    # ---------------------------------------------------------

    trail_weather[
        "freeze_thaw_day"
    ] = (
        (
            trail_weather[
                "temperature_2m_min"
            ]
            <= 32
        )
        &
        (
            trail_weather[
                "temperature_2m_max"
            ]
            > 32
        )
    ).astype(int)

    trail_weather[
        "freeze_thaw_3d"
    ] = (
        trail_weather
        .groupby(
            "trail_name"
        )[
            "freeze_thaw_day"
        ]
        .transform(
            lambda series:
                series.rolling(
                    window=3,
                    min_periods=1,
                ).sum()
        )
    )

    trail_weather[
        "freeze_thaw_7d"
    ] = (
        trail_weather
        .groupby(
            "trail_name"
        )[
            "freeze_thaw_day"
        ]
        .transform(
            lambda series:
                series.rolling(
                    window=7,
                    min_periods=1,
                ).sum()
        )
    )

    # ---------------------------------------------------------
    # SNOW
    # ---------------------------------------------------------

    trail_weather[
        "snowfall_1d"
    ] = trail_weather[
        "snowfall_sum"
    ]

    trail_weather[
        "snowfall_3d"
    ] = (
        trail_weather
        .groupby(
            "trail_name"
        )[
            "snowfall_sum"
        ]
        .transform(
            lambda series:
                series.rolling(
                    window=3,
                    min_periods=1,
                ).sum()
        )
    )

    trail_weather[
        "snowfall_7d"
    ] = (
        trail_weather
        .groupby(
            "trail_name"
        )[
            "snowfall_sum"
        ]
        .transform(
            lambda series:
                series.rolling(
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

    trail_weather[
        "days_since_precip"
    ] = (
        trail_weather
        .groupby(
            "trail_name",
            group_keys=False,
        )[
            "precipitation_sum"
        ]
        .apply(
            calculate_days_since_precip
        )
    )

    return trail_weather


# ============================================================
# MAIN
# ============================================================


def main():

    (
        weather_path,
        topography_path,
        output_path,
    ) = get_project_paths()

    # ---------------------------------------------------------
    # LOAD HISTORICAL WEATHER
    # ---------------------------------------------------------

    print(
        "Loading master historical weather..."
    )

    weather = pd.read_csv(
        weather_path,
        parse_dates=["date"],
    )

    print(
        f"Weather rows: "
        f"{len(weather):,}"
    )

    print(
        f"Weather trails: "
        f"{weather['trail_name'].nunique():,}"
    )

    print(
        f"Weather dates: "
        f"{weather['date'].nunique():,}"
    )

    # ---------------------------------------------------------
    # BASIC WEATHER VALIDATION
    # ---------------------------------------------------------

    duplicate_weather = (
        weather.duplicated(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sum()
    )

    print(
        f"Duplicate trail/date weather rows: "
        f"{duplicate_weather:,}"
    )

    if duplicate_weather > 0:

        raise RuntimeError(
            "Duplicate trail/date weather rows found."
        )

    # ---------------------------------------------------------
    # BUILD WEATHER FEATURES
    # ---------------------------------------------------------

    print()
    print(
        "Building precipitation features..."
    )

    print(
        "Building temperature features..."
    )

    print(
        "Building freeze-thaw features..."
    )

    print(
        "Building snowfall features..."
    )

    weather_features = (
        build_weather_features(
            weather
        )
    )

    # ---------------------------------------------------------
    # LOAD TOPOGRAPHY
    # ---------------------------------------------------------

    print()
    print(
        "Loading master topography features..."
    )

    topography = pd.read_csv(
        topography_path
    )

    print(
        f"Topography trails: "
        f"{len(topography):,}"
    )

    # ---------------------------------------------------------
    # DETERMINE TERRAIN COLUMNS
    # ---------------------------------------------------------
    #
    # Keep useful terrain fields while avoiding duplicated
    # final_area fields during the merge.
    # ---------------------------------------------------------

    terrain_candidates = [
        "trail_name",
        "sampled_length_miles",
        "minimum_elevation_feet",
        "maximum_elevation_feet",
        "mean_elevation_feet",
        "elevation_gain_feet",
        "elevation_loss_feet",
        "mean_slope_degrees",
        "median_slope_degrees",
        "maximum_slope_degrees",
        "north_facing_pct",
        "east_facing_pct",
        "south_facing_pct",
        "west_facing_pct",
    ]

    terrain_columns = [
        column
        for column
        in terrain_candidates
        if column in topography.columns
    ]

    if "trail_name" not in terrain_columns:

        raise RuntimeError(
            "Topography dataset is missing trail_name."
        )

    print(
        "Terrain columns being merged:"
    )

    for column in terrain_columns:

        print(
            f"  - {column}"
        )

    # ---------------------------------------------------------
    # MERGE TERRAIN
    # ---------------------------------------------------------

    print()
    print(
        "Merging terrain features..."
    )

    dataset = (
        weather_features
        .merge(
            topography[
                terrain_columns
            ],
            on="trail_name",
            how="left",
            validate="many_to_one",
        )
    )

    # ---------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------

    duplicate_rows = (
        dataset.duplicated(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sum()
    )

    unique_trails = (
        dataset[
            "trail_name"
        ]
        .nunique()
    )

    unique_dates = (
        dataset[
            "date"
        ]
        .nunique()
    )

    missing_terrain = 0

    if (
        "mean_elevation_feet"
        in dataset.columns
    ):

        missing_terrain = (
            dataset[
                "mean_elevation_feet"
            ]
            .isna()
            .sum()
        )

    missing_weather = (
        dataset[
            [
                "temperature_2m_mean",
                "precipitation_sum",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = (
        dataset
        .sort_values(
            [
                "date",
                "trail_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    dataset.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MASTER WEATHER FEATURE BUILD COMPLETE"
    )
    print("=" * 72)

    print(
        f"Rows: "
        f"{len(dataset):,}"
    )

    print(
        f"Trails: "
        f"{unique_trails:,}"
    )

    print(
        f"Dates: "
        f"{unique_dates:,}"
    )

    print(
        f"Duplicate trail/date rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"Missing weather rows: "
        f"{missing_weather:,}"
    )

    print(
        f"Missing terrain rows: "
        f"{missing_terrain:,}"
    )

    print(
        f"First date: "
        f"{dataset['date'].min().date()}"
    )

    print(
        f"Last date: "
        f"{dataset['date'].max().date()}"
    )

    print()
    print(
        f"Saved to:"
        f"\n  {output_path}"
    )

    # ---------------------------------------------------------
    # LATEST-DATE SAMPLE
    # ---------------------------------------------------------

    latest_date = (
        dataset[
            "date"
        ]
        .max()
    )

    latest = (
        dataset[
            dataset[
                "date"
            ]
            == latest_date
        ]
        .copy()
    )

    print()
    print(
        f"Latest-date sample — "
        f"{latest_date.date()}:"
    )

    print()

    preview_columns = [
        "trail_name",
        "final_area",
        "temperature_2m_mean",
        "precip_1d",
        "precip_3d",
        "precip_7d",
        "days_since_precip",
        "snowfall_3d",
        "mean_elevation_feet",
        "mean_slope_degrees",
    ]

    preview_columns = [
        column
        for column
        in preview_columns
        if column in latest.columns
    ]

    print(
        latest[
            preview_columns
        ]
        .head(25)
        .round(3)
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # FEATURE RANGES
    # ---------------------------------------------------------

    print()
    print(
        "Feature ranges:"
    )

    print(
        f"  precip_1d: "
        f"{dataset['precip_1d'].min():.3f}"
        f"–"
        f"{dataset['precip_1d'].max():.3f} in"
    )

    print(
        f"  precip_3d: "
        f"{dataset['precip_3d'].min():.3f}"
        f"–"
        f"{dataset['precip_3d'].max():.3f} in"
    )

    print(
        f"  precip_7d: "
        f"{dataset['precip_7d'].min():.3f}"
        f"–"
        f"{dataset['precip_7d'].max():.3f} in"
    )

    print(
        f"  days_since_precip: "
        f"{dataset['days_since_precip'].min()}"
        f"–"
        f"{dataset['days_since_precip'].max()}"
    )

    print(
        f"  snowfall_3d: "
        f"{dataset['snowfall_3d'].min():.3f}"
        f"–"
        f"{dataset['snowfall_3d'].max():.3f} in"
    )


if __name__ == "__main__":
    main()