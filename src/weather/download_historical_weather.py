from pathlib import Path

import pandas as pd
import requests


PARK_CITY_LATITUDE = 40.6461
PARK_CITY_LONGITUDE = -111.4980

START_DATE = "2023-01-01"
END_DATE = "2026-08-01"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    output_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_historical_weather.csv"
    )

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": PARK_CITY_LATITUDE,
        "longitude": PARK_CITY_LONGITUDE,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join(
            [
                "temperature_2m_max",
                "temperature_2m_min",
                "temperature_2m_mean",
                "precipitation_sum",
                "rain_sum",
                "snowfall_sum",
            ]
        ),
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/Denver",
    }

    print("Requesting historical Park City weather...")

    response = requests.get(
        url,
        params=params,
        timeout=60,
    )

    response.raise_for_status()

    data = response.json()

    daily = pd.DataFrame(data["daily"])

    daily["date"] = pd.to_datetime(daily["time"])
    daily = daily.drop(columns="time")

    daily = daily[
        [
            "date",
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
        ]
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily.to_csv(
        output_path,
        index=False,
    )

    print(f"Weather records downloaded: {len(daily):,}")
    print(f"First date: {daily['date'].min().date()}")
    print(f"Last date: {daily['date'].max().date()}")
    print(f"Saved to: {output_path}")

    print("\nPreview:")
    print(daily.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
