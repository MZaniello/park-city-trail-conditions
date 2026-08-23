from pathlib import Path

import pandas as pd


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def predict_condition(row):
    """
    Continuous rule-based trail rideability model.

    This is still a heuristic baseline, NOT a trained ML model.

    The goal of v2 is to:
    - avoid large threshold jumps
    - avoid excessive 100-point ties
    - allow terrain to affect drying
    - preserve interpretability
    """

    score = 100.0
    reasons = []

    precip_1d = max(0.0, float(row["precip_1d"]))
    precip_3d = max(0.0, float(row["precip_3d"]))
    precip_7d = max(0.0, float(row["precip_7d"]))

    days_since_precip = max(
        0.0,
        float(row["days_since_precip"]),
    )

    mean_temp_3d = float(row["mean_temp_3d"])
    snowfall_3d = max(
        0.0,
        float(row["snowfall_3d"]),
    )

    freeze_thaw_3d = max(
        0.0,
        float(row["freeze_thaw_3d"]),
    )

    south_pct = float(row["south_facing_pct"])
    north_pct = float(row["north_facing_pct"])

    # ---------------------------------------------------------
    # 1-DAY PRECIPITATION
    # ---------------------------------------------------------
    #
    # Strongest weather signal.
    #
    # 0.10" -> ~12 point penalty
    # 0.25" -> ~30 point penalty
    # 0.50" -> ~60 point penalty
    #

    precip_1d_penalty = min(
        65.0,
        precip_1d * 120.0,
    )

    score -= precip_1d_penalty

    if precip_1d >= 0.05:
        reasons.append(
            f"{precip_1d:.2f} in precipitation today"
        )

    # ---------------------------------------------------------
    # 3-DAY MOISTURE
    # ---------------------------------------------------------
    #
    # Smaller penalty than today's precipitation because some
    # of this moisture may already have had time to drain/dry.
    #

    precip_3d_penalty = min(
        25.0,
        precip_3d * 35.0,
    )

    score -= precip_3d_penalty

    if precip_3d >= 0.20:
        reasons.append(
            f"{precip_3d:.2f} in precipitation over 3 days"
        )

    # ---------------------------------------------------------
    # 7-DAY SATURATION
    # ---------------------------------------------------------
    #
    # Only a modest effect. This represents lingering background
    # moisture rather than immediate surface wetness.
    #

    precip_7d_penalty = min(
        10.0,
        precip_7d * 4.0,
    )

    score -= precip_7d_penalty

    if precip_7d >= 1.0:
        reasons.append(
            "elevated 7-day moisture"
        )

    # ---------------------------------------------------------
    # DRYING TIME
    # ---------------------------------------------------------

    drying_bonus = min(
        10.0,
        days_since_precip * 2.5,
    )

    score += drying_bonus

    if days_since_precip >= 2:
        reasons.append(
            f"{int(days_since_precip)} days of drying"
        )

    # ---------------------------------------------------------
    # TEMPERATURE
    # ---------------------------------------------------------
    #
    # Warm weather gets a small drying advantage.
    # Cold weather gets a modest penalty.
    #

    if mean_temp_3d > 60:

        warm_bonus = min(
            6.0,
            (mean_temp_3d - 60.0) * 0.4,
        )

        score += warm_bonus

        if warm_bonus >= 3:
            reasons.append(
                "warm drying conditions"
            )

    elif mean_temp_3d < 45:

        cold_penalty = min(
            10.0,
            (45.0 - mean_temp_3d) * 0.5,
        )

        score -= cold_penalty

        reasons.append(
            "cool conditions slowing drying"
        )

    # ---------------------------------------------------------
    # TERRAIN EXPOSURE
    # ---------------------------------------------------------
    #
    # Terrain matters more when moisture exists.
    # There is little reason to reward a south-facing trail
    # simply because a completely dry week occurred.
    #

    moisture_factor = clamp(
        precip_3d / 0.30,
        0.0,
        1.0,
    )

    south_bonus = (
        (south_pct / 100.0)
        * 5.0
        * moisture_factor
    )

    north_penalty = (
        (north_pct / 100.0)
        * 5.0
        * moisture_factor
    )

    terrain_adjustment = (
        south_bonus
        - north_penalty
    )

    score += terrain_adjustment

    if (
        moisture_factor > 0.25
        and terrain_adjustment >= 2
    ):
        reasons.append(
            "sunny exposure favors drying"
        )

    elif (
        moisture_factor > 0.25
        and terrain_adjustment <= -2
    ):
        reasons.append(
            "shaded exposure may slow drying"
        )

    # ---------------------------------------------------------
    # SNOW
    # ---------------------------------------------------------

    if snowfall_3d > 0:

        snow_penalty = min(
            70.0,
            25.0 + snowfall_3d * 20.0,
        )

        score -= snow_penalty

        reasons.append(
            "recent snowfall"
        )

    # ---------------------------------------------------------
    # FREEZE / THAW
    # ---------------------------------------------------------

    freeze_penalty = min(
        20.0,
        freeze_thaw_3d * 6.0,
    )

    score -= freeze_penalty

    if freeze_thaw_3d >= 1:
        reasons.append(
            "recent freeze-thaw activity"
        )

    # ---------------------------------------------------------
    # FINAL SCORE
    # ---------------------------------------------------------

    score = round(
        clamp(
            score,
            0.0,
            100.0,
        )
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
        / "trail_forecast_features.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "forecast_condition_predictions_v2.csv"
    )

    print("Loading forecast features...")

    dataset = pd.read_csv(
        input_path,
        parse_dates=["date"],
    )

    print(
        f"Rows loaded: {len(dataset):,}"
    )

    print(
        "Generating Baseline v2 predictions..."
    )

    predictions = dataset.apply(
        predict_condition,
        axis=1,
    )

    results = pd.concat(
        [
            dataset.reset_index(drop=True),
            predictions.reset_index(drop=True),
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # RANK WITHIN EACH DAY
    # ---------------------------------------------------------

    results["daily_rank"] = (
        results
        .groupby("date")[
            "rideability_score"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    results = results.sort_values(
        [
            "date",
            "rideability_score",
            "trail_name",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    results.to_csv(
        output_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("BASELINE V2 FORECAST PREDICTIONS")
    print("=" * 70)

    print(
        f"Rows: {len(results):,}"
    )

    print(
        f"Trails: {results['trail_name'].nunique()}"
    )

    print(
        f"Dates: {results['date'].nunique()}"
    )

    print(
        f"Saved to: {output_path}"
    )

    # ---------------------------------------------------------
    # PRINT EACH DAY
    # ---------------------------------------------------------

    for forecast_date in sorted(
        results["date"].unique()
    ):

        day = results[
            results["date"]
            == forecast_date
        ].copy()

        print()
        print("-" * 70)

        print(
            "Trail rankings — "
            f"{pd.Timestamp(forecast_date).date()}"
        )

        print("-" * 70)

        print(
            day[
                [
                    "daily_rank",
                    "trail_name",
                    "rideability_score",
                    "predicted_condition",
                    "precip_1d",
                    "precip_3d",
                    "precip_7d",
                    "days_since_precip",
                    "reason",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()