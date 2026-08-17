from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_historical_weather.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_weather_features.csv"
    )

    print("Loading historical weather...")

    weather = pd.read_csv(
        input_path,
        parse_dates=["date"],
    )

    weather = weather.sort_values("date").reset_index(drop=True)

    # ---------------------------------------------------------
    # PRECIPITATION HISTORY
    # ---------------------------------------------------------

    weather["precip_1d"] = weather["precipitation_sum"]

    weather["precip_3d"] = (
        weather["precipitation_sum"]
        .rolling(window=3, min_periods=1)
        .sum()
    )

    weather["precip_7d"] = (
        weather["precipitation_sum"]
        .rolling(window=7, min_periods=1)
        .sum()
    )

    # ---------------------------------------------------------
    # TEMPERATURE HISTORY
    # ---------------------------------------------------------

    weather["mean_temp_3d"] = (
        weather["temperature_2m_mean"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    weather["mean_temp_7d"] = (
        weather["temperature_2m_mean"]
        .rolling(window=7, min_periods=1)
        .mean()
    )

    # ---------------------------------------------------------
    # FREEZE / THAW
    #
    # True when the daily temperature crosses 32°F.
    # ---------------------------------------------------------

    weather["freeze_thaw"] = (
        (weather["temperature_2m_min"] < 32)
        & (weather["temperature_2m_max"] > 32)
    ).astype(int)

    weather["freeze_thaw_3d"] = (
        weather["freeze_thaw"]
        .rolling(window=3, min_periods=1)
        .sum()
    )

    weather["freeze_thaw_7d"] = (
        weather["freeze_thaw"]
        .rolling(window=7, min_periods=1)
        .sum()
    )

    # ---------------------------------------------------------
    # SNOW HISTORY
    # ---------------------------------------------------------

    weather["snowfall_3d"] = (
        weather["snowfall_sum"]
        .rolling(window=3, min_periods=1)
        .sum()
    )

    weather["snowfall_7d"] = (
        weather["snowfall_sum"]
        .rolling(window=7, min_periods=1)
        .sum()
    )

    # ---------------------------------------------------------
    # DAYS SINCE MEASURABLE PRECIPITATION
    # ---------------------------------------------------------

    days_since_precip = []
    days = np.nan

    for precipitation in weather["precipitation_sum"]:
        if precipitation >= 0.01:
            days = 0
        elif not np.isnan(days):
            days += 1

        days_since_precip.append(days)

    weather["days_since_precip"] = days_since_precip

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    weather.to_csv(
        output_path,
        index=False,
    )

    print(f"Weather feature rows created: {len(weather):,}")
    print(f"Saved to: {output_path}")

    print("\nNewest 10 days:")
    print(
        weather[
            [
                "date",
                "temperature_2m_mean",
                "precip_1d",
                "precip_3d",
                "precip_7d",
                "days_since_precip",
                "freeze_thaw_3d",
                "snowfall_3d",
            ]
        ]
        .tail(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()