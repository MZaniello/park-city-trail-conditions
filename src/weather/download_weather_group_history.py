from pathlib import Path
from datetime import date, timedelta
import time

import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

START_DATE = "2023-01-01"

HISTORICAL_LAG_DAYS = 5

REQUEST_TIMEOUT = 180

MAX_RETRIES = 10

# Slow and conservative because the archive API has already
# rate-limited us during the full 373-trail attempt.
WAIT_BETWEEN_GROUPS_SECONDS = 25

RATE_LIMIT_WAIT_SECONDS = 120

METERS_PER_FOOT = 0.3048


DAILY_VARIABLES = [
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

    summary_path = (
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

    progress_path = (
        project_root
        / "data"
        / "raw"
        / "weather_group_history_progress.csv"
    )

    group_history_path = (
        project_root
        / "data"
        / "raw"
        / "weather_group_historical_weather.csv"
    )

    trail_history_path = (
        project_root
        / "data"
        / "raw"
        / "master_trail_historical_weather.csv"
    )

    return (
        summary_path,
        trail_mapping_path,
        progress_path,
        group_history_path,
        trail_history_path,
    )


# ============================================================
# HELPERS
# ============================================================


def get_end_date():

    return (
        date.today()
        - timedelta(
            days=HISTORICAL_LAG_DAYS
        )
    ).isoformat()


def load_progress(
    progress_path,
):

    if not progress_path.exists():

        return pd.DataFrame(
            columns=GROUP_OUTPUT_COLUMNS
        )

    progress = pd.read_csv(
        progress_path,
        parse_dates=["date"],
    )

    print(
        f"Existing progress rows: "
        f"{len(progress):,}"
    )

    print(
        f"Completed weather groups: "
        f"{progress['weather_group_id'].nunique():,}"
    )

    return progress


def save_progress(
    progress,
    progress_path,
):

    progress = (
        progress
        .drop_duplicates(
            subset=[
                "weather_group_id",
                "date",
            ]
        )
        .sort_values(
            [
                "weather_group_id",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    progress_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress.to_csv(
        progress_path,
        index=False,
    )


def request_group_history(
    group,
    start_date,
    end_date,
):

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

        "start_date":
            start_date,

        "end_date":
            end_date,

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
    }

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.get(
                OPEN_METEO_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:

                return response.json()

            if response.status_code == 429:

                wait_seconds = min(
                    600,
                    RATE_LIMIT_WAIT_SECONDS
                    * (
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
                    180,
                    15
                    * (
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

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                raise

            wait_seconds = min(
                180,
                15
                * (
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
        "Open-Meteo request failed "
        "after all retries."
    )


def parse_group_response(
    group,
    response_data,
):

    if "daily" not in response_data:

        raise RuntimeError(
            "No daily weather returned for "
            f"{group['weather_group_id']}."
        )

    daily = pd.DataFrame(
        response_data[
            "daily"
        ]
    )

    if daily.empty:

        raise RuntimeError(
            "Empty weather response for "
            f"{group['weather_group_id']}."
        )

    daily = daily.rename(
        columns={
            "time":
                "date"
        }
    )

    daily["date"] = (
        pd.to_datetime(
            daily["date"]
        )
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

    return daily[
        GROUP_OUTPUT_COLUMNS
    ]


def map_group_weather_to_trails(
    group_history,
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

    trail_weather = (
        trail_mapping[
            trail_columns
        ]
        .merge(
            group_history,
            on="weather_group_id",
            how="left",
            validate="many_to_many",
        )
    )

    trail_weather = trail_weather.rename(
        columns={
            "latitude":
                "trail_latitude",

            "longitude":
                "trail_longitude",

            "mean_elevation_feet":
                "trail_elevation_feet",
        }
    )

    trail_weather = trail_weather[
        [
            "date",
            "trail_name",
            "final_area",
            "weather_group_id",
            "trail_latitude",
            "trail_longitude",
            "trail_elevation_feet",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
        ]
    ].copy()

    return (
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


# ============================================================
# MAIN
# ============================================================


def main():

    (
        summary_path,
        trail_mapping_path,
        progress_path,
        group_history_path,
        trail_history_path,
    ) = get_project_paths()

    end_date = get_end_date()

    # ---------------------------------------------------------
    # LOAD WEATHER GROUPS
    # ---------------------------------------------------------

    print(
        "Loading weather group summary..."
    )

    groups = pd.read_csv(
        summary_path
    )

    print(
        f"Weather groups: "
        f"{len(groups):,}"
    )

    # ---------------------------------------------------------
    # LOAD TRAIL MAPPING
    # ---------------------------------------------------------

    print(
        "Loading trail-to-weather mapping..."
    )

    trail_mapping = pd.read_csv(
        trail_mapping_path
    )

    print(
        f"Trails mapped: "
        f"{len(trail_mapping):,}"
    )

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    duplicate_groups = (
        groups[
            "weather_group_id"
        ]
        .duplicated()
        .sum()
    )

    missing_group_ids = (
        set(
            trail_mapping[
                "weather_group_id"
            ]
        )
        - set(
            groups[
                "weather_group_id"
            ]
        )
    )

    print()
    print(
        f"Duplicate weather groups: "
        f"{duplicate_groups:,}"
    )

    print(
        f"Missing referenced groups: "
        f"{len(missing_group_ids):,}"
    )

    if duplicate_groups > 0:

        raise RuntimeError(
            "Duplicate weather group IDs found."
        )

    if missing_group_ids:

        raise RuntimeError(
            "Trail mapping references weather groups "
            "that do not exist in the summary."
        )

    # ---------------------------------------------------------
    # LOAD PROGRESS
    # ---------------------------------------------------------

    progress = load_progress(
        progress_path
    )

    completed_groups = set(
        progress[
            "weather_group_id"
        ].unique()
    )

    remaining = groups[
        ~groups[
            "weather_group_id"
        ].isin(
            completed_groups
        )
    ].copy()

    print()
    print(
        f"Historical start date: "
        f"{START_DATE}"
    )

    print(
        f"Historical end date: "
        f"{end_date}"
    )

    print(
        f"Completed groups: "
        f"{len(completed_groups):,}"
    )

    print(
        f"Remaining groups: "
        f"{len(remaining):,}"
    )

    # ---------------------------------------------------------
    # DOWNLOAD
    # ---------------------------------------------------------

    frames = []

    if not progress.empty:

        frames.append(
            progress
        )

    total_remaining = len(
        remaining
    )

    for number, (
        _,
        group,
    ) in enumerate(
        remaining.iterrows(),
        start=1,
    ):

        group_id = (
            group[
                "weather_group_id"
            ]
        )

        print()
        print(
            f"[{number}/{total_remaining}] "
            f"Downloading {group_id} "
            f"({int(group['trail_count'])} trails)..."
        )

        response_data = (
            request_group_history(
                group,
                START_DATE,
                end_date,
            )
        )

        group_weather = (
            parse_group_response(
                group,
                response_data,
            )
        )

        frames.append(
            group_weather
        )

        current_progress = (
            pd.concat(
                frames,
                ignore_index=True,
            )
        )

        save_progress(
            current_progress,
            progress_path,
        )

        print(
            f"  Saved "
            f"{len(group_weather):,} rows."
        )

        print(
            "  Progress written to disk."
        )

        if number < total_remaining:

            print(
                f"  Waiting "
                f"{WAIT_BETWEEN_GROUPS_SECONDS} "
                "seconds..."
            )

            time.sleep(
                WAIT_BETWEEN_GROUPS_SECONDS
            )

    # ---------------------------------------------------------
    # FINAL GROUP HISTORY
    # ---------------------------------------------------------

    if frames:

        group_history = (
            pd.concat(
                frames,
                ignore_index=True,
            )
        )

    else:

        group_history = progress

    group_history = (
        group_history
        .drop_duplicates(
            subset=[
                "weather_group_id",
                "date",
            ]
        )
        .sort_values(
            [
                "weather_group_id",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    group_history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_history.to_csv(
        group_history_path,
        index=False,
    )

    # ---------------------------------------------------------
    # MAP WEATHER TO ALL 373 TRAILS
    # ---------------------------------------------------------

    print()
    print(
        "Mapping weather-group history "
        "back to individual trails..."
    )

    trail_weather = (
        map_group_weather_to_trails(
            group_history,
            trail_mapping,
        )
    )

    trail_weather.to_csv(
        trail_history_path,
        index=False,
    )

    # ---------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------

    expected_dates = len(
        pd.date_range(
            START_DATE,
            end_date,
            freq="D",
        )
    )

    expected_group_rows = (
        len(groups)
        * expected_dates
    )

    expected_trail_rows = (
        len(trail_mapping)
        * expected_dates
    )

    duplicate_group_rows = (
        group_history.duplicated(
            subset=[
                "weather_group_id",
                "date",
            ]
        )
        .sum()
    )

    duplicate_trail_rows = (
        trail_weather.duplicated(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sum()
    )

    missing_trail_weather = (
        trail_weather[
            "temperature_2m_mean"
        ]
        .isna()
        .sum()
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "WEATHER GROUP HISTORICAL DOWNLOAD COMPLETE"
    )
    print("=" * 72)

    print(
        f"Weather groups: "
        f"{group_history['weather_group_id'].nunique():,}"
    )

    print(
        f"Group-history rows: "
        f"{len(group_history):,}"
    )

    print(
        f"Expected group-history rows: "
        f"{expected_group_rows:,}"
    )

    print(
        f"Group duplicate rows: "
        f"{duplicate_group_rows:,}"
    )

    print()
    print(
        f"Trails mapped: "
        f"{trail_weather['trail_name'].nunique():,}"
    )

    print(
        f"Trail-history rows: "
        f"{len(trail_weather):,}"
    )

    print(
        f"Expected trail-history rows: "
        f"{expected_trail_rows:,}"
    )

    print(
        f"Trail duplicate rows: "
        f"{duplicate_trail_rows:,}"
    )

    print(
        f"Missing weather rows: "
        f"{missing_trail_weather:,}"
    )

    print(
        f"First date: "
        f"{trail_weather['date'].min().date()}"
    )

    print(
        f"Last date: "
        f"{trail_weather['date'].max().date()}"
    )

    print()
    print(
        f"Saved weather-group history to:"
        f"\n  {group_history_path}"
    )

    print()
    print(
        f"Saved 373-trail historical weather to:"
        f"\n  {trail_history_path}"
    )

    print()
    print(
        f"Resume file:"
        f"\n  {progress_path}"
    )


if __name__ == "__main__":
    main()