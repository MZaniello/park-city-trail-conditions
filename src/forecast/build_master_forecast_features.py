from pathlib import Path

import numpy as np
import pandas as pd


PRECIP_THRESHOLD_INCHES = 0.01

FORECAST_DAYS = 7


def get_project_paths():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    recent_forecast_path = (
        project_root
        / "data"
        / "raw"
        / "master_trail_forecast.csv"
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
        / "master_trail_forecast_features.csv"
    )

    return (
        recent_forecast_path,
        topography_path,
        output_path,
    )


def calculate_days_since_precip(
    precipitation,
):

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

        if value >= PRECIP_THRESHOLD_INCHES:

            dry_days = 0

        else:

            dry_days += 1

        result[index] = dry_days

    return pd.Series(
        result,
        index=precipitation.index,
    )


def validate_consecutive_dates(data):
    """
    Fail if any trail has a calendar-date gap.
    """

    print(
        "Checking for date gaps..."
    )

    gap_records = []

    for trail_name, group in data.groupby(
        "trail_name"
    ):

        dates = (
            group[
                "date"
            ]
            .drop_duplicates()
            .sort_values()
        )

        differences = (
            dates.diff()
        )

        bad = differences[
            differences
            != pd.Timedelta(
                days=1
            )
        ].dropna()

        if not bad.empty:

            for index in bad.index:

                current_date = (
                    dates.loc[index]
                )

                previous_position = (
                    dates.index
                    .get_loc(index)
                    - 1
                )

                previous_date = (
                    dates.iloc[
                        previous_position
                    ]
                )

                gap_records.append(
                    {
                        "trail_name":
                            trail_name,

                        "previous_date":
                            previous_date,

                        "current_date":
                            current_date,

                        "gap_days":
                            (
                                current_date
                                - previous_date
                            ).days,
                    }
                )

    if gap_records:

        gaps = pd.DataFrame(
            gap_records
        )

        print()
        print(
            "DATE GAP ERROR"
        )

        print(
            gaps.head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            f"Found {len(gaps)} date gaps. "
            "Rolling features would be invalid."
        )

    print(
        "Date continuity check passed."
    )


def build_features(data):

    data = (
        data
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

    data[
        "precip_1d"
    ] = data[
        "precipitation_sum"
    ]

    data[
        "precip_3d"
    ] = (
        data
        .groupby(
            "trail_name"
        )[
            "precipitation_sum"
        ]
        .transform(
            lambda s:
                s.rolling(
                    3,
                    min_periods=1,
                ).sum()
        )
    )

    data[
        "precip_7d"
    ] = (
        data
        .groupby(
            "trail_name"
        )[
            "precipitation_sum"
        ]
        .transform(
            lambda s:
                s.rolling(
                    7,
                    min_periods=1,
                ).sum()
        )
    )

    data[
        "mean_temp_3d"
    ] = (
        data
        .groupby(
            "trail_name"
        )[
            "temperature_2m_mean"
        ]
        .transform(
            lambda s:
                s.rolling(
                    3,
                    min_periods=1,
                ).mean()
        )
    )

    data[
        "mean_temp_7d"
    ] = (
        data
        .groupby(
            "trail_name"
        )[
            "temperature_2m_mean"
        ]
        .transform(
            lambda s:
                s.rolling(
                    7,
                    min_periods=1,
                ).mean()
        )
    )

    data[
        "freeze_thaw_day"
    ] = (
        (
            data[
                "temperature_2m_min"
            ]
            <= 32
        )
        &
        (
            data[
                "temperature_2m_max"
            ]
            > 32
        )
    ).astype(int)

    data[
        "freeze_thaw_3d"
    ] = (
        data
        .groupby(
            "trail_name"
        )[
            "freeze_thaw_day"
        ]
        .transform(
            lambda s:
                s.rolling(
                    3,
                    min_periods=1,
                ).sum()
        )
    )

    data[
        "freeze_thaw_7d"
    ] = (
        data
        .groupby(
            "trail_name"
        )[
            "freeze_thaw_day"
        ]
        .transform(
            lambda s:
                s.rolling(
                    7,
                    min_periods=1,
                ).sum()
        )
    )

    data[
        "snowfall_1d"
    ] = data[
        "snowfall_sum"
    ]

    data[
        "snowfall_3d"
    ] = (
        data
        .groupby(
            "trail_name"
        )[
            "snowfall_sum"
        ]
        .transform(
            lambda s:
                s.rolling(
                    3,
                    min_periods=1,
                ).sum()
        )
    )

    data[
        "snowfall_7d"
    ] = (
        data
        .groupby(
            "trail_name"
        )[
            "snowfall_sum"
        ]
        .transform(
            lambda s:
                s.rolling(
                    7,
                    min_periods=1,
                ).sum()
        )
    )

    print(
        "Calculating days since precipitation..."
    )

    data[
        "days_since_precip"
    ] = (
        data
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

    return data


def main():

    (
        recent_forecast_path,
        topography_path,
        output_path,
    ) = get_project_paths()

    print(
        "Loading recent + forecast weather..."
    )

    weather = pd.read_csv(
        recent_forecast_path,
        parse_dates=["date"],
    )

    print(
        f"Rows: "
        f"{len(weather):,}"
    )

    print(
        f"Trails: "
        f"{weather['trail_name'].nunique():,}"
    )

    print(
        f"Dates: "
        f"{weather['date'].nunique():,}"
    )

    print(
        f"First date: "
        f"{weather['date'].min().date()}"
    )

    print(
        f"Last date: "
        f"{weather['date'].max().date()}"
    )

    duplicate_rows = (
        weather
        .duplicated(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sum()
    )

    if duplicate_rows > 0:

        raise RuntimeError(
            f"Found "
            f"{duplicate_rows} duplicate "
            "trail/date rows."
        )

    validate_consecutive_dates(
        weather
    )

    print()
    print(
        "Building rolling weather features..."
    )

    weather = build_features(
        weather
    )

    # ---------------------------------------------------------
    # DEFINE ACTUAL FORECAST WINDOW
    # ---------------------------------------------------------
    #
    # Since the file now includes past_days, the final
    # FORECAST_DAYS calendar dates are the true forecast dates.
    # ---------------------------------------------------------

    unique_dates = sorted(
        weather[
            "date"
        ]
        .unique()
    )

    if len(unique_dates) < FORECAST_DAYS:

        raise RuntimeError(
            "Not enough dates returned "
            "to isolate forecast window."
        )

    forecast_dates = (
        unique_dates[
            -FORECAST_DAYS:
        ]
    )

    forecast_features = (
        weather[
            weather[
                "date"
            ]
            .isin(
                forecast_dates
            )
        ]
        .copy()
    )

    print()
    print(
        "Forecast dates retained:"
    )

    for forecast_date in forecast_dates:

        print(
            f"  - "
            f"{pd.Timestamp(forecast_date).date()}"
        )

    print()
    print(
        "Loading terrain features..."
    )

    topography = pd.read_csv(
        topography_path
    )

    terrain_columns = [
        "trail_name",
        "sampled_length_miles",
        "minimum_elevation_feet",
        "maximum_elevation_feet",
        "mean_elevation_feet",
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
        in terrain_columns
        if column
        in topography.columns
    ]

    forecast_features = (
        forecast_features
        .merge(
            topography[
                terrain_columns
            ],
            on="trail_name",
            how="left",
            validate="many_to_one",
        )
    )

    duplicate_forecast_rows = (
        forecast_features
        .duplicated(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sum()
    )

    missing_weather = (
        forecast_features[
            [
                "temperature_2m_mean",
                "precip_1d",
                "precip_3d",
                "precip_7d",
                "days_since_precip",
                "mean_temp_3d",
                "snowfall_3d",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    missing_terrain = (
        forecast_features[
            "mean_elevation_feet"
        ]
        .isna()
        .sum()
    )

    unique_trails = (
        forecast_features[
            "trail_name"
        ]
        .nunique()
    )

    forecast_date_count = (
        forecast_features[
            "date"
        ]
        .nunique()
    )

    expected_rows = (
        unique_trails
        * forecast_date_count
    )

    forecast_features = (
        forecast_features
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_features.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 72)
    print(
        "MASTER FORECAST FEATURE BUILD COMPLETE"
    )
    print("=" * 72)

    print(
        f"Rows: "
        f"{len(forecast_features):,}"
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
        f"{forecast_date_count:,}"
    )

    print(
        f"Duplicate trail/date rows: "
        f"{duplicate_forecast_rows:,}"
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
        f"First forecast date: "
        f"{forecast_features['date'].min().date()}"
    )

    print(
        f"Last forecast date: "
        f"{forecast_features['date'].max().date()}"
    )

    print()
    print(
        f"Saved to:"
        f"\n  {output_path}"
    )

    first_date = (
        forecast_features[
            "date"
        ]
        .min()
    )

    sample = (
        forecast_features[
            forecast_features[
                "date"
            ]
            == first_date
        ]
        .head(25)
    )

    print()
    print(
        f"First-date feature sample — "
        f"{first_date.date()}:"
    )

    print()

    print(
        sample[
            [
                "trail_name",
                "final_area",
                "temperature_2m_mean",
                "precip_1d",
                "precip_3d",
                "precip_7d",
                "days_since_precip",
                "mean_temp_3d",
                "freeze_thaw_3d",
                "snowfall_3d",
                "mean_elevation_feet",
                "south_facing_pct",
            ]
        ]
        .round(3)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()