from pathlib import Path

import pandas as pd


def baseline_score(row):
    """
    Reproduce the existing rule-based baseline model.

    This is NOT a trained machine-learning model.
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
        reasons.append(
            "heavy precipitation in last 24 hours"
        )

    elif row["precip_1d"] >= 0.20:
        score -= 35
        reasons.append(
            "moderate precipitation in last 24 hours"
        )

    elif row["precip_1d"] >= 0.05:
        score -= 15
        reasons.append(
            "light precipitation in last 24 hours"
        )

    # ---------------------------------------------------------
    # RECENT SATURATION — LAST 3 DAYS
    # ---------------------------------------------------------

    if row["precip_3d"] >= 1.00:
        score -= 25
        reasons.append(
            "very wet previous 3 days"
        )

    elif row["precip_3d"] >= 0.50:
        score -= 15
        reasons.append(
            "wet previous 3 days"
        )

    elif row["precip_3d"] >= 0.20:
        score -= 7
        reasons.append(
            "some recent moisture"
        )

    # ---------------------------------------------------------
    # DRYING TIME
    # ---------------------------------------------------------

    if row["days_since_precip"] >= 3:
        score += 5
        reasons.append(
            "several days of drying"
        )

    # ---------------------------------------------------------
    # TEMPERATURE
    # ---------------------------------------------------------

    if row["mean_temp_3d"] >= 70:
        score += 5
        reasons.append(
            "warm drying conditions"
        )

    elif row["mean_temp_3d"] < 40:
        score -= 10
        reasons.append(
            "cold conditions slowing drying"
        )

    # ---------------------------------------------------------
    # FREEZE / THAW
    # ---------------------------------------------------------

    if row["freeze_thaw_3d"] >= 2:
        score -= 15
        reasons.append(
            "repeated freeze-thaw cycles"
        )

    return score, reasons


def terrain_adjustment(row):
    """
    Apply a SMALL heuristic adjustment based on terrain exposure.

    These values are deliberately conservative because we do not
    yet have enough real condition labels to calibrate them.
    """

    adjustment = 0
    reasons = []

    south_pct = row["south_facing_pct"]
    north_pct = row["north_facing_pct"]

    # ---------------------------------------------------------
    # SOUTH-FACING DRYING ADVANTAGE
    # ---------------------------------------------------------

    if south_pct >= 60:
        adjustment += 4
        reasons.append(
            "strong south-facing drying exposure"
        )

    elif south_pct >= 40:
        adjustment += 2
        reasons.append(
            "moderate south-facing drying exposure"
        )

    # ---------------------------------------------------------
    # NORTH-FACING DRYING PENALTY
    # ---------------------------------------------------------

    if north_pct >= 50:
        adjustment -= 4
        reasons.append(
            "strong north-facing shade exposure"
        )

    elif north_pct >= 30:
        adjustment -= 2
        reasons.append(
            "moderate north-facing shade exposure"
        )

    return adjustment, reasons


def predict_condition(row):
    """
    Generate an enhanced heuristic rideability prediction.
    """

    base_score, reasons = baseline_score(row)

    terrain_score, terrain_reasons = (
        terrain_adjustment(row)
    )

    score = base_score + terrain_score

    reasons.extend(
        terrain_reasons
    )

    # ---------------------------------------------------------
    # KEEP SCORE IN RANGE
    # ---------------------------------------------------------

    score = max(
        0,
        min(
            100,
            score,
        ),
    )

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
        reasons.append(
            "no major weather concerns"
        )

    return pd.Series(
        {
            "rideability_score":
                score,

            "predicted_condition":
                condition,

            "reason":
                "; ".join(reasons),
        }
    )


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    input_path = (
        project_root
        / "data"
        / "processed"
        / "trail_forecast_features.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "forecast_condition_predictions.csv"
    )

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    print(
        "Loading forecast features..."
    )

    forecast = pd.read_csv(
        input_path,
        parse_dates=[
            "date"
        ],
    )

    print(
        f"Forecast rows: "
        f"{len(forecast):,}"
    )

    # ---------------------------------------------------------
    # PREDICT
    # ---------------------------------------------------------

    print(
        "Generating condition predictions..."
    )

    predictions = forecast.apply(
        predict_condition,
        axis=1,
    )

    results = pd.concat(
        [
            forecast.reset_index(
                drop=True
            ),
            predictions.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # RANK TRAILS WITHIN EACH DAY
    # ---------------------------------------------------------

    results["daily_rank"] = (
        results
        .groupby("date")[
            "rideability_score"
        ]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    results = results.sort_values(
        [
            "date",
            "daily_rank",
            "trail_name",
        ]
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    duplicate_rows = (
        results.duplicated(
            subset=[
                "date",
                "trail_name",
            ]
        ).sum()
    )

    print()
    print(
        "=" * 60
    )

    print(
        "FORECAST CONDITION PREDICTIONS COMPLETE"
    )

    print(
        "=" * 60
    )

    print(
        f"Rows: "
        f"{len(results):,}"
    )

    print(
        f"Trails: "
        f"{results['trail_name'].nunique()}"
    )

    print(
        f"Dates: "
        f"{results['date'].nunique()}"
    )

    print(
        f"Duplicate trail/date rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    # ---------------------------------------------------------
    # PRINT DAILY RANKINGS
    # ---------------------------------------------------------

    for forecast_date in sorted(
        results["date"].unique()
    ):

        day = results[
            results["date"]
            == forecast_date
        ].copy()

        print()
        print(
            "-" * 60
        )

        print(
            f"Trail rankings — "
            f"{pd.Timestamp(forecast_date).date()}"
        )

        print(
            "-" * 60
        )

        print(
            day[
                [
                    "daily_rank",
                    "trail_name",
                    "rideability_score",
                    "predicted_condition",
                    "precip_1d",
                    "precip_3d",
                    "days_since_precip",
                    "south_facing_pct",
                    "north_facing_pct",
                    "reason",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()