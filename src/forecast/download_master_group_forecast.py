from pathlib import Path
import time

import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

FORECAST_DAYS = 7
PAST_DAYS = 10

REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
WAIT_BETWEEN_GROUPS_SECONDS = 1

METERS_PER_FOOT = 0.3048


DAILY_VARIABLES = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
]


GROUP_OUTPUT_COLUMNS = [
    "date",
    "weather_group_id",
    "representative_latitude",
    "representative_longitude",
    "representative_elevation_feet",
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
]


# ============================================================
# PATHS
# ============================================================

def get_project_paths():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    group_summary_path = (
        project_root
        / "data"
        / "processed"
        / "weather_group_summary.csv"
    )

    trail_mapping_path = (
        project_root
        / "data"
        / "processed"
        / "master_weather_groups.csv"
    )

    group_forecast_path = (
        project_root
        / "data"
        / "raw"
        / "weather_group_forecast.csv"
    )

    trail_forecast_path = (
        project_root
        / "data"
        / "raw"
        / "master_trail_forecast.csv"
    )

    return (
        group_summary_path,
        trail_mapping_path,
        group_forecast_path,
        trail_forecast_path,
    )


# ============================================================
# REQUEST
# ============================================================

def request_group_forecast(group):

    elevation_meters = (
        float(
            group[
                "representative_elevation_feet"
            ]
        )
        * METERS_PER_FOOT
    )

    params = {
        "latitude":
            float(
                group[
                    "representative_latitude"
                ]
            ),

        "longitude":
            float(
                group[
                    "representative_longitude"
                ]
            ),

        "elevation":
            round(
                elevation_meters,
                1,
            ),

        "daily":
            ",".join(
                DAILY_VARIABLES
            ),

        "temperature_unit":
            "fahrenheit",

        "precipitation_unit":
            "inch",

        "timezone":
            "America/Denver",

        "forecast_days":
            FORECAST_DAYS,

        "past_days":
            PAST_DAYS,
    }

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.get(
                FORECAST_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:

                wait_seconds = min(
                    300,
                    15 * (
                        attempt + 1
                    ),
                )

                print(
                    "  Rate limited (429). "
                    f"Waiting {wait_seconds} seconds..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            if response.status_code in {
                500,
                502,
                503,
                504,
            }:

                wait_seconds = min(
                    120,
                    5 * (
                        2 ** attempt
                    ),
                )

                print(
                    f"  Temporary API response "
                    f"{response.status_code}. "
                    f"Waiting {wait_seconds} seconds..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            response.raise_for_status()

        except requests.RequestException as error:

            if attempt == MAX_RETRIES - 1:
                raise

            wait_seconds = min(
                120,
                5 * (
                    2 ** attempt
                ),
            )

            print(
                f"  Request error: {error}"
            )

            print(
                f"  Waiting {wait_seconds} seconds..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "Forecast request failed "
        "after all retries."
    )


# ============================================================
# PARSE
# ============================================================

def parse_group_forecast(
    group,
    response_data,
):

    if "daily" not in response_data:

        raise RuntimeError(
            "No daily forecast returned for "
            f"{group['weather_group_id']}."
        )

    daily = pd.DataFrame(
        response_data["daily"]
    )

    if daily.empty:

        raise RuntimeError(
            "Empty forecast returned for "
            f"{group['weather_group_id']}."
        )

    daily = daily.rename(
        columns={
            "time": "date",
        }
    )

    daily["date"] = pd.to_datetime(
        daily["date"]
    )

    daily[
        "weather_group_id"
    ] = group[
        "weather_group_id"
    ]

    daily[
        "representative_latitude"
    ] = float(
        group[
            "representative_latitude"
        ]
    )

    daily[
        "representative_longitude"
    ] = float(
        group[
            "representative_longitude"
        ]
    )

    daily[
        "representative_elevation_feet"
    ] = float(
        group[
            "representative_elevation_feet"
        ]
    )

    missing_columns = [
        column
        for column in GROUP_OUTPUT_COLUMNS
        if column not in daily.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Open-Meteo response is missing "
            "expected columns: "
            + ", ".join(
                missing_columns
            )
        )

    return daily[
        GROUP_OUTPUT_COLUMNS
    ]


# ============================================================
# MAP GROUP WEATHER TO TRAILS
# ============================================================

def map_forecast_to_trails(
    group_forecast,
    trail_mapping,
):

    trail_columns = [
        "trail_name",
        "final_area",
        "latitude",
        "longitude",
        "mean_elevation_feet",
        "weather_group_id",
    ]

    forecast = (
        trail_mapping[
            trail_columns
        ]
        .merge(
            group_forecast,
            on="weather_group_id",
            how="left",
            validate="many_to_many",
        )
    )

    forecast = forecast.rename(
        columns={
            "latitude":
                "trail_latitude",

            "longitude":
                "trail_longitude",

            "mean_elevation_feet":
                "trail_elevation_feet",
        }
    )

    forecast = forecast[
        [
            "date",
            "trail_name",
            "final_area",
            "weather_group_id",
            "trail_latitude",
            "trail_longitude",
            "trail_elevation_feet",
            "weather_code",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
        ]
    ]

    return (
        forecast
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


# ============================================================
# MAIN
# ============================================================

def main():

    (
        group_summary_path,
        trail_mapping_path,
        group_forecast_path,
        trail_forecast_path,
    ) = get_project_paths()

    print(
        "Loading weather groups..."
    )

    groups = pd.read_csv(
        group_summary_path
    )

    print(
        f"Weather groups: "
        f"{len(groups):,}"
    )

    print(
        "Loading trail mapping..."
    )

    trail_mapping = pd.read_csv(
        trail_mapping_path
    )

    print(
        f"Trails: "
        f"{len(trail_mapping):,}"
    )

    duplicate_groups = (
        groups[
            "weather_group_id"
        ]
        .duplicated()
        .sum()
    )

    duplicate_trails = (
        trail_mapping[
            "trail_name"
        ]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate groups: "
        f"{duplicate_groups:,}"
    )

    print(
        f"Duplicate trails: "
        f"{duplicate_trails:,}"
    )

    if duplicate_groups > 0:

        raise RuntimeError(
            "Duplicate weather group IDs found."
        )

    if duplicate_trails > 0:

        raise RuntimeError(
            "Duplicate trail names found."
        )

    print()
    print(
        f"Downloading "
        f"{PAST_DAYS} past days + "
        f"{FORECAST_DAYS} forecast days "
        "with weather codes..."
    )

    frames = []

    total_groups = len(
        groups
    )

    for number, (
        _,
        group,
    ) in enumerate(
        groups.iterrows(),
        start=1,
    ):

        group_id = (
            group[
                "weather_group_id"
            ]
        )

        print(
            f"[{number}/{total_groups}] "
            f"{group_id} "
            f"({int(group['trail_count'])} trails)",
            flush=True,
        )

        response_data = (
            request_group_forecast(
                group
            )
        )

        group_data = (
            parse_group_forecast(
                group,
                response_data,
            )
        )

        frames.append(
            group_data
        )

        if number < total_groups:

            time.sleep(
                WAIT_BETWEEN_GROUPS_SECONDS
            )

    print()
    print(
        "Combining weather-group data..."
    )

    group_forecast = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "date",
                "weather_group_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    group_dates = (
        group_forecast[
            "date"
        ]
        .nunique()
    )

    expected_group_rows = (
        len(groups)
        * group_dates
    )

    duplicate_group_rows = (
        group_forecast
        .duplicated(
            subset=[
                "weather_group_id",
                "date",
            ]
        )
        .sum()
    )

    missing_group_weather = (
        group_forecast[
            [
                "weather_code",
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

    group_forecast_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_forecast.to_csv(
        group_forecast_path,
        index=False,
    )

    print(
        "Mapping weather to 373 trails..."
    )

    trail_forecast = (
        map_forecast_to_trails(
            group_forecast,
            trail_mapping,
        )
    )

    trail_dates = (
        trail_forecast[
            "date"
        ]
        .nunique()
    )

    expected_trail_rows = (
        len(trail_mapping)
        * trail_dates
    )

    duplicate_trail_rows = (
        trail_forecast
        .duplicated(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sum()
    )

    missing_trail_weather = (
        trail_forecast[
            [
                "weather_code",
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

    trail_forecast.to_csv(
        trail_forecast_path,
        index=False,
    )

    print()
    print("=" * 72)
    print(
        "MASTER RECENT + FORECAST DOWNLOAD COMPLETE"
    )
    print("=" * 72)

    print(
        f"Weather groups: "
        f"{group_forecast['weather_group_id'].nunique():,}"
    )

    print(
        f"Total dates returned: "
        f"{group_dates:,}"
    )

    print(
        f"Group rows: "
        f"{len(group_forecast):,}"
    )

    print(
        f"Expected group rows: "
        f"{expected_group_rows:,}"
    )

    print(
        f"Duplicate group/date rows: "
        f"{duplicate_group_rows:,}"
    )

    print(
        f"Missing group weather rows: "
        f"{missing_group_weather:,}"
    )

    print()
    print(
        f"Trails: "
        f"{trail_forecast['trail_name'].nunique():,}"
    )

    print(
        f"Trail rows: "
        f"{len(trail_forecast):,}"
    )

    print(
        f"Expected trail rows: "
        f"{expected_trail_rows:,}"
    )

    print(
        f"Duplicate trail/date rows: "
        f"{duplicate_trail_rows:,}"
    )

    print(
        f"Missing trail weather rows: "
        f"{missing_trail_weather:,}"
    )

    print(
        f"First returned date: "
        f"{trail_forecast['date'].min().date()}"
    )

    print(
        f"Last returned date: "
        f"{trail_forecast['date'].max().date()}"
    )

    print()
    print(
        "Weather-code distribution:"
    )

    print(
        trail_forecast[
            "weather_code"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        f"Saved weather-group data to:"
        f"\n  {group_forecast_path}"
    )

    print()
    print(
        f"Saved trail recent+forecast data to:"
        f"\n  {trail_forecast_path}"
    )


if __name__ == "__main__":
    main()