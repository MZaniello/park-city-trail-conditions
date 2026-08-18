from pathlib import Path
from datetime import date, timedelta
import time

import pandas as pd
import requests


START_DATE = date(2023, 1, 1)

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
]

OUTPUT_COLUMNS = [
    "date",
    "trail_name",
    "trail_latitude",
    "trail_longitude",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
]


def download_weather(
    latitude,
    longitude,
    start_date,
    end_date,
):
    """
    Download weather for one trail and one date range.

    Returns a DataFrame containing the daily Open-Meteo data.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/Denver",
    }

    max_retries = 5

    for attempt in range(max_retries):

        response = requests.get(
            url,
            params=params,
            timeout=60,
        )

        if response.status_code == 200:
            data = response.json()

            if "daily" not in data:
                raise RuntimeError(
                    "Open-Meteo response did not contain daily data."
                )

            return pd.DataFrame(data["daily"])

        if response.status_code == 429:

            wait_seconds = 10 * (attempt + 1)

            print(
                f"  Rate limited. "
                f"Waiting {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)
            continue

        response.raise_for_status()

    raise RuntimeError(
        "Open-Meteo request failed after multiple retries."
    )


def load_existing_weather(output_path):
    """
    Load the existing weather dataset if one exists.
    """

    if not output_path.exists():
        print("No existing weather file found.")
        print("A full historical download will be performed.")

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    print("Loading existing weather data...")

    weather = pd.read_csv(output_path)

    if weather.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    weather["date"] = pd.to_datetime(
        weather["date"],
        errors="coerce",
    )

    invalid_dates = weather["date"].isna().sum()

    if invalid_dates:
        print(
            f"Removing {invalid_dates:,} row(s) "
            "with invalid dates."
        )

        weather = weather[
            weather["date"].notna()
        ].copy()

    print(
        f"Existing rows: {len(weather):,}"
    )

    if not weather.empty:
        print(
            "Existing date range: "
            f"{weather['date'].min().date()} "
            f"through "
            f"{weather['date'].max().date()}"
        )

    return weather


def get_next_start_date(
    existing_weather,
    trail_name,
):
    """
    Determine the first missing date after the latest
    stored observation for a trail.
    """

    if existing_weather.empty:
        return START_DATE

    trail_weather = existing_weather[
        existing_weather["trail_name"]
        == trail_name
    ]

    if trail_weather.empty:
        return START_DATE

    latest_date = trail_weather[
        "date"
    ].max()

    if pd.isna(latest_date):
        return START_DATE

    return (
        latest_date.date()
        + timedelta(days=1)
    )


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    locations_path = (
        project_root
        / "data"
        / "processed"
        / "trail_locations.csv"
    )

    output_path = (
        project_root
        / "data"
        / "raw"
        / "trail_historical_weather.csv"
    )

    today = date.today()

    # ---------------------------------------------------------
    # LOAD TRAILS
    # ---------------------------------------------------------

    print("Loading trail locations...")

    trails = pd.read_csv(
        locations_path
    )

    print(
        f"Trails to process: {len(trails)}"
    )

    print(
        f"Target end date: {today}"
    )

    # ---------------------------------------------------------
    # LOAD EXISTING WEATHER
    # ---------------------------------------------------------

    existing_weather = (
        load_existing_weather(
            output_path
        )
    )

    new_weather_frames = []

    trails_downloaded = 0
    trails_current = 0

    # ---------------------------------------------------------
    # PROCESS EACH TRAIL
    # ---------------------------------------------------------

    for number, trail in trails.iterrows():

        trail_name = trail[
            "trail_name"
        ]

        start_date = get_next_start_date(
            existing_weather,
            trail_name,
        )

        print()
        print(
            f"[{number + 1}/{len(trails)}] "
            f"{trail_name}"
        )

        # Trail already contains weather through today.
        if start_date > today:

            print(
                "  Already current. "
                "No download needed."
            )

            trails_current += 1
            continue

        print(
            f"  Downloading "
            f"{start_date} through {today}..."
        )

        weather = download_weather(
            latitude=trail["latitude"],
            longitude=trail["longitude"],
            start_date=start_date,
            end_date=today,
        )

        if weather.empty:

            print(
                "  No new weather rows returned."
            )

            continue

        weather = weather.rename(
            columns={
                "time": "date"
            }
        )

        weather["date"] = pd.to_datetime(
            weather["date"]
        )

        weather["trail_name"] = (
            trail_name
        )

        weather["trail_latitude"] = (
            trail["latitude"]
        )

        weather["trail_longitude"] = (
            trail["longitude"]
        )

        weather = weather[
            OUTPUT_COLUMNS
        ]

        new_weather_frames.append(
            weather
        )

        trails_downloaded += 1

        print(
            f"  New rows: {len(weather):,}"
        )

        # Small pause to reduce rate-limit risk.
        time.sleep(2)

    # ---------------------------------------------------------
    # COMBINE OLD + NEW DATA
    # ---------------------------------------------------------

    print()
    print("Combining weather datasets...")

    frames = []

    if not existing_weather.empty:
        frames.append(
            existing_weather
        )

    frames.extend(
        new_weather_frames
    )

    if frames:

        combined = pd.concat(
            frames,
            ignore_index=True,
        )

    else:

        combined = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    if not combined.empty:

        combined["date"] = pd.to_datetime(
            combined["date"],
            errors="coerce",
        )

        combined = combined[
            combined["date"].notna()
        ].copy()

        # -----------------------------------------------------
        # REMOVE DUPLICATES
        # -----------------------------------------------------

        rows_before = len(combined)

        combined = combined.drop_duplicates(
            subset=[
                "date",
                "trail_name",
            ],
            keep="last",
        )

        duplicates_removed = (
            rows_before
            - len(combined)
        )

        # -----------------------------------------------------
        # SORT
        # -----------------------------------------------------

        combined = combined.sort_values(
            [
                "date",
                "trail_name",
            ]
        ).reset_index(
            drop=True
        )

        combined = combined[
            OUTPUT_COLUMNS
        ]

    else:

        duplicates_removed = 0

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    duplicate_rows = (
        combined.duplicated(
            subset=[
                "date",
                "trail_name",
            ]
        ).sum()
        if not combined.empty
        else 0
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print("WEATHER UPDATE COMPLETE")
    print("=" * 60)

    print(
        f"Trails checked: "
        f"{len(trails)}"
    )

    print(
        f"Trails already current: "
        f"{trails_current}"
    )

    print(
        f"Trails downloaded: "
        f"{trails_downloaded}"
    )

    new_rows = sum(
        len(frame)
        for frame
        in new_weather_frames
    )

    print(
        f"New rows downloaded: "
        f"{new_rows:,}"
    )

    print(
        f"Duplicate rows removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Total rows: "
        f"{len(combined):,}"
    )

    if not combined.empty:

        print(
            f"Unique trails: "
            f"{combined['trail_name'].nunique()}"
        )

        print(
            f"Unique dates: "
            f"{combined['date'].nunique():,}"
        )

        print(
            f"Duplicate trail/date rows: "
            f"{duplicate_rows:,}"
        )

        print(
            f"First date: "
            f"{combined['date'].min().date()}"
        )

        print(
            f"Last date: "
            f"{combined['date'].max().date()}"
        )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
    