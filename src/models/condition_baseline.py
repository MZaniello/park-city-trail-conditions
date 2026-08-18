from pathlib import Path

import pandas as pd


TARGET_DATE = "2026-08-01"


def predict_condition(row):
    """
    Estimate trail rideability using recent weather.

    This is a rule-based baseline, NOT a trained ML model.
    """

    score = 100
    reasons = []

    # ---------------------------------------------------------
    # SNOW
    # ---------------------------------------------------------

    if row["snowfall_3d"] >= 2:
        score -= 70
        reasons.append("significant recent snow")

    elif row["snowfall_3d"] > 0:
        score -= 35
        reasons.append("recent snow")

    # ---------------------------------------------------------
    # PRECIPITATION — LAST 24 HOURS
    # ---------------------------------------------------------

    if row["precip_1d"] >= 0.50:
        score -= 55
        reasons.append("heavy precipitation in last 24 hours")

    elif row["precip_1d"] >= 0.20:
        score -= 35
        reasons.append("moderate precipitation in last 24 hours")

    elif row["precip_1d"] >= 0.05:
        score -= 15
        reasons.append("light precipitation in last 24 hours")

    # ---------------------------------------------------------
    # RECENT SATURATION — LAST 3 DAYS
    # ---------------------------------------------------------

    if row["precip_3d"] >= 1.00:
        score -= 25
        reasons.append("very wet previous 3 days")

    elif row["precip_3d"] >= 0.50:
        score -= 15
        reasons.append("wet previous 3 days")

    elif row["precip_3d"] >= 0.20:
        score -= 7
        reasons.append("some recent moisture")

    # ---------------------------------------------------------
    # DRYING TIME
    # ---------------------------------------------------------

    if row["days_since_precip"] >= 3:
        score += 5
        reasons.append("several days of drying")

    # ---------------------------------------------------------
    # TEMPERATURE
    # ---------------------------------------------------------

    if row["mean_temp_3d"] >= 70:
        score += 5
        reasons.append("warm drying conditions")

    elif row["mean_temp_3d"] < 40:
        score -= 10
        reasons.append("cold conditions slowing drying")

    # ---------------------------------------------------------
    # FREEZE / THAW
    # ---------------------------------------------------------

    if row["freeze_thaw_3d"] >= 2:
        score -= 15
        reasons.append("repeated freeze-thaw cycles")

    # Keep score between 0 and 100.
    score = max(0, min(100, score))

    # ---------------------------------------------------------
    # CONDITION LABEL
    # ---------------------------------------------------------

    if score >= 90:
        condition = "IDEAL"

    elif score >= 75:
        condition = "GOOD"

    elif score >= 55:
        condition = "MARGINAL"

    elif score >= 30:
        condition = "WET"

    else:
        condition = "POOR"

    if not reasons:
        reasons.append("no major weather concerns")

    return pd.Series(
        {
            "rideability_score": score,
            "predicted_condition": condition,
            "reason": "; ".join(reasons),
        }
    )


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "processed"
        / "trail_modeling_dataset.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "baseline_condition_predictions.csv"
    )

    print("Loading modeling dataset...")

    dataset = pd.read_csv(
        input_path,
        parse_dates=["date"],
    )

    target_date = pd.Timestamp(TARGET_DATE)

    today = dataset[
        dataset["date"] == target_date
    ].copy()

    if today.empty:
        raise ValueError(
            f"No observations found for {TARGET_DATE}"
        )

    print(
        f"Generating predictions for "
        f"{TARGET_DATE}..."
    )

    predictions = today.apply(
        predict_condition,
        axis=1,
    )

    results = pd.concat(
        [
            today[
                [
                    "date",
                    "trail_name",
                    "temperature_2m_mean",
                    "precip_1d",
                    "precip_3d",
                    "precip_7d",
                    "days_since_precip",
                    "snowfall_3d",
                    "minimum_elevation_feet",
                    "maximum_elevation_feet",
                ]
            ].reset_index(drop=True),
            predictions.reset_index(drop=True),
        ],
        axis=1,
    )

    results = results.sort_values(
        [
            "rideability_score",
            "trail_name",
        ],
        ascending=[
            False,
            True,
        ],
    )

    results.to_csv(
        output_path,
        index=False,
    )

    print("\nPredicted trail conditions:\n")

    print(
        results[
            [
                "trail_name",
                "rideability_score",
                "predicted_condition",
                "precip_1d",
                "precip_3d",
                "days_since_precip",
                "reason",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()