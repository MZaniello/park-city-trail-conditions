from pathlib import Path

import pandas as pd


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def estimate_moisture(row):
    """
    Estimate current surface moisture from recent precipitation,
    drying time, temperature, and terrain exposure.

    This is a heuristic index, not a physical soil-moisture
    measurement and not a trained ML model.

    Approximate interpretation:

        < 0.12   very dry / dusty
        0.12-0.28 dry
        0.28-0.55 ideal moisture
        0.55-0.75 damp
        0.75-1.00 wet
        > 1.00   muddy / saturated
    """

    precip_1d = max(
        0.0,
        float(row["precip_1d"]),
    )

    precip_3d = max(
        0.0,
        float(row["precip_3d"]),
    )

    precip_7d = max(
        0.0,
        float(row["precip_7d"]),
    )

    days_since_precip = max(
        0.0,
        float(row["days_since_precip"]),
    )

    mean_temp_3d = float(
        row["mean_temp_3d"]
    )

    south_pct = float(
        row["south_facing_pct"]
    )

    north_pct = float(
        row["north_facing_pct"]
    )

    # ---------------------------------------------------------
    # MOISTURE INPUT
    # ---------------------------------------------------------
    #
    # Today's rain matters most.
    # Older precipitation matters progressively less.
    #

    moisture = (
        precip_1d * 2.4
        + precip_3d * 0.9
        + precip_7d * 0.12
    )

    # ---------------------------------------------------------
    # DRYING FROM TIME
    # ---------------------------------------------------------

    time_drying = min(
        0.40,
        days_since_precip * 0.08,
    )

    moisture -= time_drying

    # ---------------------------------------------------------
    # TEMPERATURE DRYING
    # ---------------------------------------------------------

    if mean_temp_3d > 60:

        temperature_drying = min(
            0.20,
            (mean_temp_3d - 60)
            * 0.012,
        )

        moisture -= temperature_drying

    elif mean_temp_3d < 45:

        cold_retention = min(
            0.15,
            (45 - mean_temp_3d)
            * 0.01,
        )

        moisture += cold_retention

    # ---------------------------------------------------------
    # TERRAIN EXPOSURE
    # ---------------------------------------------------------
    #
    # South-facing terrain dries somewhat faster.
    # North-facing terrain retains somewhat more moisture.
    #

    south_drying = (
        south_pct / 100.0
    ) * 0.12

    north_retention = (
        north_pct / 100.0
    ) * 0.10

    moisture -= south_drying
    moisture += north_retention

    return max(
        0.0,
        moisture,
    )


def classify_surface(
    moisture,
    snowfall_3d,
):
    """
    Convert estimated moisture into a rider-facing surface state.
    """

    if snowfall_3d >= 0.25:
        return "SNOW"

    if moisture < 0.12:
        return "DUSTY"

    if moisture < 0.28:
        return "DRY"

    if moisture < 0.55:
        return "IDEAL"

    if moisture < 0.75:
        return "DAMP"

    if moisture < 1.00:
        return "WET"

    return "MUDDY"


def base_rideability(surface):
    """
    Assign rideability separately from surface quality.

    Dust is undesirable but generally still rideable.
    Mud receives a much larger penalty because riding may be
    unpleasant and can damage trails.
    """

    scores = {
        "IDEAL": 100.0,
        "DRY": 88.0,
        "DUSTY": 76.0,
        "DAMP": 84.0,
        "WET": 58.0,
        "MUDDY": 25.0,
        "SNOW": 35.0,
    }

    return scores[surface]


def moisture_quality_adjustment(
    moisture,
    surface,
):
    """
    Add variation within each surface class.

    This prevents every trail classified as IDEAL, DRY, etc.
    from receiving exactly the same score.
    """

    # Ideal target is roughly the middle of the hero-dirt band.
    ideal_target = 0.40

    if surface == "IDEAL":

        distance = abs(
            moisture - ideal_target
        )

        adjustment = -(
            distance * 30.0
        )

    elif surface == "DRY":

        # Wetter end of DRY is generally preferable.
        adjustment = (
            moisture - 0.12
        ) * 25.0

    elif surface == "DUSTY":

        # Near the DRY threshold is better than extremely dry.
        adjustment = min(
            5.0,
            moisture * 35.0,
        )

    elif surface == "DAMP":

        # Drier end of DAMP is preferable.
        adjustment = -(
            moisture - 0.55
        ) * 25.0

    elif surface == "WET":

        # Rideability falls rapidly as saturation increases.
        adjustment = -(
            moisture - 0.75
        ) * 35.0

    elif surface == "MUDDY":

        # Increasing saturation should make already-muddy
        # conditions progressively worse.
        adjustment = -min(
            20.0,
            (moisture - 1.0)
            * 20.0,
        )

    else:
        adjustment = 0.0

    return adjustment


def predict_condition(row):
    """
    Baseline v3.

    Predict:
        1. estimated moisture
        2. surface state
        3. rideability

    This remains a heuristic model until condition reports
    provide enough ground-truth observations for calibration.
    """

    snowfall_3d = max(
        0.0,
        float(row["snowfall_3d"]),
    )

    freeze_thaw_3d = max(
        0.0,
        float(row["freeze_thaw_3d"]),
    )

    mean_temp_3d = float(
        row["mean_temp_3d"]
    )

    precip_1d = max(
        0.0,
        float(row["precip_1d"]),
    )

    days_since_precip = max(
        0.0,
        float(row["days_since_precip"]),
    )

    # ---------------------------------------------------------
    # ESTIMATE SURFACE MOISTURE
    # ---------------------------------------------------------

    moisture = estimate_moisture(
        row
    )

    # ---------------------------------------------------------
    # CLASSIFY SURFACE
    # ---------------------------------------------------------

    surface = classify_surface(
        moisture,
        snowfall_3d,
    )

    # ---------------------------------------------------------
    # BASE RIDEABILITY
    # ---------------------------------------------------------

    score = base_rideability(
        surface
    )

    score += moisture_quality_adjustment(
        moisture,
        surface,
    )

    reasons = []

    # ---------------------------------------------------------
    # SURFACE REASON
    # ---------------------------------------------------------

    if surface == "IDEAL":

        reasons.append(
            "moisture near hero-dirt range"
        )

    elif surface == "DRY":

        reasons.append(
            "surface likely dry but firm"
        )

    elif surface == "DUSTY":

        reasons.append(
            "extended drying may produce dust or loose soil"
        )

    elif surface == "DAMP":

        reasons.append(
            "moisture slightly above ideal range"
        )

    elif surface == "WET":

        reasons.append(
            "surface moisture likely affecting riding"
        )

    elif surface == "MUDDY":

        reasons.append(
            "high moisture creates muddy or saturated risk"
        )

    elif surface == "SNOW":

        reasons.append(
            "recent snowfall affects riding"
        )

    # ---------------------------------------------------------
    # ACTIVE PRECIPITATION PENALTY
    # ---------------------------------------------------------
    #
    # A trail can have an otherwise good moisture balance but
    # rain falling today still creates uncertainty / wet spots.
    #

    if precip_1d >= 0.30:

        score -= 12

        reasons.append(
            "substantial precipitation today"
        )

    elif precip_1d >= 0.15:

        score -= 7

        reasons.append(
            "moderate precipitation today"
        )

    elif precip_1d >= 0.05:

        score -= 3

        reasons.append(
            "light precipitation today"
        )

    # ---------------------------------------------------------
    # EXTENDED DRY SPELL
    # ---------------------------------------------------------

    if (
        days_since_precip >= 5
        and surface in {
            "DUSTY",
            "DRY",
        }
    ):

        dry_penalty = min(
            8.0,
            (
                days_since_precip - 4
            ) * 1.5,
        )

        score -= dry_penalty

        reasons.append(
            f"{int(days_since_precip)} days since measurable moisture"
        )

    # ---------------------------------------------------------
    # FREEZE / THAW
    # ---------------------------------------------------------

    if freeze_thaw_3d >= 1:

        freeze_penalty = min(
            18.0,
            freeze_thaw_3d * 5.0,
        )

        score -= freeze_penalty

        reasons.append(
            "recent freeze-thaw activity"
        )

    # ---------------------------------------------------------
    # VERY COLD CONDITIONS
    # ---------------------------------------------------------

    if mean_temp_3d < 35:

        score -= 8

        reasons.append(
            "cold temperatures reduce rideability"
        )

    # ---------------------------------------------------------
    # SNOW AMOUNT
    # ---------------------------------------------------------

    if surface == "SNOW":

        additional_snow_penalty = min(
            25.0,
            snowfall_3d * 8.0,
        )

        score -= additional_snow_penalty

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

    moisture = round(
        moisture,
        3,
    )

    # ---------------------------------------------------------
    # GENERAL RIDEABILITY BAND
    # ---------------------------------------------------------

    if score >= 90:
        rideability = "EXCELLENT"

    elif score >= 75:
        rideability = "GOOD"

    elif score >= 55:
        rideability = "FAIR"

    elif score >= 35:
        rideability = "POOR"

    else:
        rideability = "AVOID"

    return pd.Series(
        {
            "estimated_moisture": moisture,
            "surface_state": surface,
            "rideability_score": score,
            "rideability": rideability,
            "reason": "; ".join(reasons),
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
        / "forecast_condition_predictions_v3.csv"
    )

    print(
        "Loading forecast features..."
    )

    dataset = pd.read_csv(
        input_path,
        parse_dates=[
            "date"
        ],
    )

    print(
        f"Rows loaded: {len(dataset):,}"
    )

    print(
        "Generating Baseline v3 predictions..."
    )

    predictions = dataset.apply(
        predict_condition,
        axis=1,
    )

    results = pd.concat(
        [
            dataset.reset_index(
                drop=True
            ),
            predictions.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )

    # ---------------------------------------------------------
    # DAILY RANKING
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
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    results.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "BASELINE V3 — HERO DIRT MODEL"
    )
    print("=" * 72)

    print(
        f"Rows: {len(results):,}"
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
        f"Saved to: {output_path}"
    )

    # ---------------------------------------------------------
    # DAILY OUTPUT
    # ---------------------------------------------------------

    for forecast_date in sorted(
        results["date"].unique()
    ):

        day = results[
            results["date"]
            == forecast_date
        ].copy()

        print()
        print("-" * 72)

        print(
            "Trail rankings — "
            f"{pd.Timestamp(forecast_date).date()}"
        )

        print("-" * 72)

        print(
            day[
                [
                    "daily_rank",
                    "trail_name",
                    "rideability_score",
                    "rideability",
                    "surface_state",
                    "estimated_moisture",
                    "precip_1d",
                    "precip_3d",
                    "days_since_precip",
                    "reason",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()