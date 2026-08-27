from pathlib import Path

import pandas as pd


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def estimate_moisture(row):
    """
    Estimate retained surface moisture using non-overlapping
    precipitation windows plus temperature / terrain retention.

    This is a heuristic moisture index, not measured soil moisture.

    Key idea:
    - today's rain matters most
    - rain from the previous 2 days still matters substantially
    - rain from days 4-7 matters only modestly
    - temperature, aspect, and elevation modify retention
    """

    precip_1d = max(
        0.0,
        float(row["precip_1d"]),
    )

    precip_3d = max(
        precip_1d,
        float(row["precip_3d"]),
    )

    precip_7d = max(
        precip_3d,
        float(row["precip_7d"]),
    )

    mean_temp_3d = float(
        row["mean_temp_3d"]
    )

    days_since_precip = max(
        0.0,
        float(row["days_since_precip"]),
    )

    south_pct = float(
        row["south_facing_pct"]
    )

    north_pct = float(
        row["north_facing_pct"]
    )

    minimum_elevation = float(
        row["minimum_elevation_feet"]
    )

    maximum_elevation = float(
        row["maximum_elevation_feet"]
    )

    mean_elevation = (
        minimum_elevation
        + maximum_elevation
    ) / 2.0

    # ---------------------------------------------------------
    # NON-OVERLAPPING PRECIPITATION BUCKETS
    # ---------------------------------------------------------

    rain_today = precip_1d

    rain_previous_2_days = max(
        0.0,
        precip_3d - precip_1d,
    )

    rain_days_4_to_7 = max(
        0.0,
        precip_7d - precip_3d,
    )

    # ---------------------------------------------------------
    # BASE MOISTURE INPUT
    # ---------------------------------------------------------

    moisture = (
        rain_today * 1.35
        + rain_previous_2_days * 0.70
        + rain_days_4_to_7 * 0.18
    )

    # ---------------------------------------------------------
    # TEMPERATURE RETENTION FACTOR
    # ---------------------------------------------------------

    if mean_temp_3d >= 75:
        temperature_factor = 0.82

    elif mean_temp_3d >= 65:
        temperature_factor = 0.90

    elif mean_temp_3d >= 55:
        temperature_factor = 0.97

    elif mean_temp_3d >= 45:
        temperature_factor = 1.04

    else:
        temperature_factor = 1.12

    moisture *= temperature_factor

    # ---------------------------------------------------------
    # ASPECT RETENTION FACTOR
    # ---------------------------------------------------------

    south_effect = (
        south_pct / 100.0
    ) * 0.10

    north_effect = (
        north_pct / 100.0
    ) * 0.10

    aspect_factor = (
        1.0
        - south_effect
        + north_effect
    )

    moisture *= aspect_factor

    # ---------------------------------------------------------
    # ELEVATION RETENTION FACTOR
    # ---------------------------------------------------------

    elevation_factor = 1.0

    if mean_elevation >= 9000:
        elevation_factor = 1.10

    elif mean_elevation >= 8000:
        elevation_factor = 1.06

    elif mean_elevation >= 7500:
        elevation_factor = 1.03

    elif mean_elevation < 7000:
        elevation_factor = 0.97

    moisture *= elevation_factor

    # ---------------------------------------------------------
    # DRY-SPELL DECAY
    # ---------------------------------------------------------

    if days_since_precip >= 1:

        decay_factor = 0.90 ** min(
            days_since_precip,
            7,
        )

        moisture *= decay_factor

    return max(
        0.0,
        moisture,
    )


def classify_surface(
    moisture,
    snowfall_3d,
):
    """
    Convert moisture index into surface state.
    """

    if snowfall_3d >= 0.25:
        return "SNOW"

    if moisture < 0.06:
        return "DUSTY"

    if moisture < 0.14:
        return "DRY"

    if moisture < 0.32:
        return "IDEAL"

    if moisture < 0.50:
        return "DAMP"

    if moisture < 0.75:
        return "WET"

    return "MUDDY"


def base_rideability(surface):
    """
    Surface state and rideability are intentionally separate.

    Dusty trails are still highly rideable compared with
    wet or muddy trails, but hero dirt receives the highest score.
    """

    scores = {
        "IDEAL": 100.0,
        "DRY": 86.0,
        "DUSTY": 74.0,
        "DAMP": 86.0,
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
    Add within-category variation so trails with the same
    surface classification do not automatically receive
    identical scores.
    """

    ideal_target = 0.22

    if surface == "IDEAL":

        distance = abs(
            moisture - ideal_target
        )

        adjustment = -(
            distance * 35.0
        )

    elif surface == "DRY":

        # Wetter end of DRY is generally preferable.
        normalized = (
            moisture - 0.06
        ) / (
            0.14 - 0.06
        )

        adjustment = normalized * 4.0

    elif surface == "DUSTY":

        # Slightly dusty is preferable to extremely dry.
        normalized = clamp(
            moisture / 0.06,
            0.0,
            1.0,
        )

        adjustment = normalized * 4.0

    elif surface == "DAMP":

        # Drier end of DAMP is generally preferable.
        normalized = (
            moisture - 0.32
        ) / (
            0.50 - 0.32
        )

        adjustment = -(
            normalized * 8.0
        )

    elif surface == "WET":

        # Rideability falls more rapidly as saturation increases.
        normalized = (
            moisture - 0.50
        ) / (
            0.75 - 0.50
        )

        adjustment = -(
            normalized * 18.0
        )

    elif surface == "MUDDY":

        additional = max(
            0.0,
            moisture - 0.75,
        )

        adjustment = -min(
            22.0,
            additional * 28.0,
        )

    else:
        adjustment = 0.0

    return adjustment


def predict_condition(row):
    """
    Baseline v3.2 hero-dirt rideability model.

    Predicts:
        1. estimated retained moisture
        2. surface state
        3. rideability score
        4. rider-facing rideability category

    This remains a heuristic model until enough real-world
    condition reports exist to calibrate it.
    """

    snowfall_3d = max(
        0.0,
        float(row["snowfall_3d"]),
    )

    freeze_thaw_3d = max(
        0.0,
        float(row["freeze_thaw_3d"]),
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
    # ESTIMATE RETAINED MOISTURE
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
    # SURFACE DESCRIPTION
    # ---------------------------------------------------------

    if surface == "IDEAL":

        reasons.append(
            "moisture near hero-dirt range"
        )

    elif surface == "DRY":

        reasons.append(
            "surface likely dry but still firm"
        )

    elif surface == "DUSTY":

        reasons.append(
            "low retained moisture may produce dust or loose soil"
        )

    elif surface == "DAMP":

        reasons.append(
            "surface slightly wetter than ideal"
        )

    elif surface == "WET":

        reasons.append(
            "surface moisture likely affecting traction"
        )

    elif surface == "MUDDY":

        reasons.append(
            "high retained moisture creates muddy or saturated risk"
        )

    elif surface == "SNOW":

        reasons.append(
            "recent snowfall affects riding"
        )

    # ---------------------------------------------------------
    # ACTIVE PRECIPITATION
    # ---------------------------------------------------------

    if precip_1d >= 0.50:

        score -= 15

        reasons.append(
            "heavy precipitation today"
        )

    elif precip_1d >= 0.25:

        score -= 10

        reasons.append(
            "substantial precipitation today"
        )

    elif precip_1d >= 0.10:

        score -= 5

        reasons.append(
            "moderate precipitation today"
        )

    elif precip_1d >= 0.05:

        score -= 2

        reasons.append(
            "light precipitation today"
        )

    # ---------------------------------------------------------
    # EXTENDED DRY SPELL
    # ---------------------------------------------------------

    if (
        surface in {
            "DUSTY",
            "DRY",
        }
        and days_since_precip >= 5
    ):

        penalty = min(
            7.0,
            (
                days_since_precip - 4
            ) * 1.25,
        )

        score -= penalty

        reasons.append(
            f"{int(days_since_precip)} days since measurable precipitation"
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
    # SNOW
    # ---------------------------------------------------------

    if surface == "SNOW":

        score -= min(
            25.0,
            snowfall_3d * 8.0,
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

    moisture = round(
        moisture,
        3,
    )

    # ---------------------------------------------------------
    # RIDEABILITY CATEGORY
    # ---------------------------------------------------------
    #
    # EXCELLENT is deliberately reserved for genuinely
    # exceptional riding conditions rather than simply
    # "not wet."
    #

    if score >= 93:
        rideability = "EXCELLENT"

    elif score >= 78:
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
        / "forecast_condition_predictions_v3_2.csv"
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
        "Generating Baseline v3.2 predictions..."
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
        "BASELINE V3.2 — HERO DIRT RIDEABILITY CALIBRATION"
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