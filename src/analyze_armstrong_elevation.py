from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SMOOTHING_WINDOW = 5


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "processed"
        / "armstrong_elevation_samples.csv"
    )

    chart_path = (
        project_root
        / "outputs"
        / "figures"
        / "armstrong_elevation_profile.png"
    )

    summary_path = (
        project_root
        / "data"
        / "processed"
        / "armstrong_elevation_summary.csv"
    )

    print("Loading Armstrong elevation samples...")
    samples = pd.read_csv(input_path)

    required_columns = {
        "distance_meters",
        "distance_miles",
        "elevation_meters",
        "elevation_feet",
    }

    missing_columns = required_columns - set(samples.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    samples = samples.sort_values("distance_meters").reset_index(drop=True)

    # Smooth small DEM fluctuations before calculating gain and loss.
    samples["smoothed_elevation_meters"] = (
        samples["elevation_meters"]
        .rolling(
            window=SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    elevation_change = samples[
        "smoothed_elevation_meters"
    ].diff()

    elevation_gain_meters = elevation_change.clip(lower=0).sum()
    elevation_loss_meters = -elevation_change.clip(upper=0).sum()

    total_distance_meters = samples["distance_meters"].max()

    net_elevation_change_meters = (
        samples["smoothed_elevation_meters"].iloc[-1]
        - samples["smoothed_elevation_meters"].iloc[0]
    )

    average_net_grade_percent = (
        net_elevation_change_meters
        / total_distance_meters
        * 100
    )

    min_elevation_feet = samples["elevation_feet"].min()
    max_elevation_feet = samples["elevation_feet"].max()

    summary = pd.DataFrame(
        {
            "trail_name": ["Armstrong"],
            "distance_miles": [
                samples["distance_miles"].max()
            ],
            "minimum_elevation_feet": [
                min_elevation_feet
            ],
            "maximum_elevation_feet": [
                max_elevation_feet
            ],
            "elevation_gain_feet": [
                elevation_gain_meters * 3.28084
            ],
            "elevation_loss_feet": [
                elevation_loss_meters * 3.28084
            ],
            "net_elevation_change_feet": [
                net_elevation_change_meters * 3.28084
            ],
            "average_net_grade_percent": [
                average_net_grade_percent
            ],
        }
    )

    summary = summary.round(
        {
            "distance_miles": 2,
            "minimum_elevation_feet": 0,
            "maximum_elevation_feet": 0,
            "elevation_gain_feet": 0,
            "elevation_loss_feet": 0,
            "net_elevation_change_feet": 0,
            "average_net_grade_percent": 1,
        }
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    chart_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(summary_path, index=False)

    plt.figure(figsize=(10, 5))

    plt.plot(
        samples["distance_miles"],
        samples["elevation_feet"],
        label="Raw DEM elevation",
        alpha=0.45,
    )

    plt.plot(
        samples["distance_miles"],
        samples["smoothed_elevation_meters"] * 3.28084,
        label="Smoothed elevation",
        linewidth=2,
    )

    plt.xlabel("Distance (miles)")
    plt.ylabel("Elevation (feet)")
    plt.title("Armstrong Trail Elevation Profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()

    print("\nArmstrong elevation summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved chart to: {chart_path}")


if __name__ == "__main__":
    main()