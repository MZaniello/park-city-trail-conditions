from pathlib import Path
from datetime import date
import time

import pandas as pd
import requests


START_DATE = "2023-01-01"

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
]


def download_weather(latitude, longitude, end_date):
    """Download historical weather for one trail location with retries."""

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": START_DATE,
        "end_date": end_date,
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


def main():
    project_root = Path(__file__).resolve().parents[2]

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

    # Open-Meteo's archive endpoint is updated continuously,
    # so use today's date instead of a hard-coded cutoff.
    end_date = date.today().isoformat()

    print("Loading trail locations...")

    trails = pd.read_csv(locations_path)

    print(f"Trails to process: {len(trails)}")
    print(f"Weather start date: {START_DATE}")
    print(f"Weather end date: {end_date}")

    all_weather = []

    for number, trail in trails.iterrows():

        trail_name = trail["trail_name"]

        print(
            f"[{number + 1}/{len(trails)}] "
            f"Downloading weather for {trail_name}..."
        )

        weather = download_weather(
            trail["latitude"],
            trail["longitude"],
            end_date,
        )

        weather["trail_name"] = trail_name
        weather["trail_latitude"] = trail["latitude"]
        weather["trail_longitude"] = trail["longitude"]

        all_weather.append(weather)

        # Slow requests slightly to reduce rate-limit problems.
        time.sleep(2)

    print("\nCombining weather datasets...")

    combined = pd.concat(
        all_weather,
        ignore_index=True,
    )

    combined = combined.rename(
        columns={"time": "date"}
    )

    combined["date"] = pd.to_datetime(
        combined["date"]
    )

    combined = combined[
        [
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
    ]

    combined = combined.sort_values(
        ["date", "trail_name"]
    ).reset_index(drop=True)

    duplicate_rows = combined.duplicated(
        subset=[
            "date",
            "trail_name",
        ]
    ).sum()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        output_path,
        index=False,
    )

    print("\nDownload complete!")
    print(f"Rows created: {len(combined):,}")
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
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()