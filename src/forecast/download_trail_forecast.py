from pathlib import Path
import time

import pandas as pd
import requests


FORECAST_DAYS = 7

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


def download_forecast(
    latitude,
    longitude,
):
    """
    Download a 7-day daily forecast for one trail location.
    """

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": FORECAST_DAYS,
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
                    "Open-Meteo response did not contain daily forecast data."
                )

            return pd.DataFrame(
                data["daily"]
            )

        if response.status_code == 429:

            wait_seconds = 10 * (
                attempt + 1
            )

            print(
                f"  Rate limited. "
                f"Waiting {wait_seconds} seconds..."
            )

            time.sleep(
                wait_seconds
            )

            continue

        response.raise_for_status()

    raise RuntimeError(
        "Open-Meteo forecast request failed "
        "after multiple retries."
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
        / "trail_forecast.csv"
    )

    # ---------------------------------------------------------
    # LOAD TRAILS
    # ---------------------------------------------------------

    print(
        "Loading trail locations..."
    )

    trails = pd.read_csv(
        locations_path
    )

    print(
        f"Trails to process: "
        f"{len(trails)}"
    )

    print(
        f"Forecast horizon: "
        f"{FORECAST_DAYS} days"
    )

    all_forecasts = []

    # ---------------------------------------------------------
    # DOWNLOAD EACH TRAIL
    # ---------------------------------------------------------

    for number, trail in trails.iterrows():

        trail_name = trail[
            "trail_name"
        ]

        print(
            f"[{number + 1}/{len(trails)}] "
            f"Downloading forecast for "
            f"{trail_name}..."
        )

        forecast = download_forecast(
            latitude=trail["latitude"],
            longitude=trail["longitude"],
        )

        forecast = forecast.rename(
            columns={
                "time": "date"
            }
        )

        forecast["date"] = (
            pd.to_datetime(
                forecast["date"]
            )
        )

        forecast["trail_name"] = (
            trail_name
        )

        forecast[
            "trail_latitude"
        ] = trail["latitude"]

        forecast[
            "trail_longitude"
        ] = trail["longitude"]

        forecast = forecast[
            OUTPUT_COLUMNS
        ]

        all_forecasts.append(
            forecast
        )

        # Small pause to reduce rate limiting.
        time.sleep(1)

    # ---------------------------------------------------------
    # COMBINE
    # ---------------------------------------------------------

    print()
    print(
        "Combining forecasts..."
    )

    combined = pd.concat(
        all_forecasts,
        ignore_index=True,
    )

    combined = combined.sort_values(
        [
            "date",
            "trail_name",
        ]
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    duplicate_rows = (
        combined.duplicated(
            subset=[
                "date",
                "trail_name",
            ]
        ).sum()
    )

    expected_rows = (
        len(trails)
        * FORECAST_DAYS
    )

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
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "FORECAST DOWNLOAD COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Rows created: "
        f"{len(combined):,}"
    )

    print(
        f"Expected rows: "
        f"{expected_rows:,}"
    )

    print(
        f"Unique trails: "
        f"{combined['trail_name'].nunique()}"
    )

    print(
        f"Unique dates: "
        f"{combined['date'].nunique()}"
    )

    print(
        f"Duplicate trail/date rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"First forecast date: "
        f"{combined['date'].min().date()}"
    )

    print(
        f"Last forecast date: "
        f"{combined['date'].max().date()}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    # ---------------------------------------------------------
    # PREVIEW
    # ---------------------------------------------------------

    print()
    print(
        "First forecast date:"
    )

    first_date = (
        combined["date"].min()
    )

    preview = combined[
        combined["date"]
        == first_date
    ][
        [
            "trail_name",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
        ]
    ]

    print(
        preview.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
    