from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.condition_baseline_v3_2 import predict_condition

TERRAIN_PROFILES = {
    "low_south": {
        "minimum_elevation_feet": 6500,
        "maximum_elevation_feet": 7000,
        "north_facing_pct": 10,
        "south_facing_pct": 60,
    },
    "mid_mixed": {
        "minimum_elevation_feet": 7400,
        "maximum_elevation_feet": 8200,
        "north_facing_pct": 30,
        "south_facing_pct": 30,
    },
    "high_north": {
        "minimum_elevation_feet": 9000,
        "maximum_elevation_feet": 9800,
        "north_facing_pct": 65,
        "south_facing_pct": 10,
    },
}

SCENARIOS = [
    {"scenario": "bone_dry_10_days", "precip_1d": 0.00, "precip_3d": 0.00, "precip_7d": 0.00, "days_since_precip": 10, "mean_temp_3d": 74, "snowfall_3d": 0.0, "freeze_thaw_3d": 0},
    {"scenario": "very_dry_5_days", "precip_1d": 0.00, "precip_3d": 0.00, "precip_7d": 0.05, "days_since_precip": 5, "mean_temp_3d": 72, "snowfall_3d": 0.0, "freeze_thaw_3d": 0},
    {"scenario": "light_recent_moisture", "precip_1d": 0.02, "precip_3d": 0.08, "precip_7d": 0.15, "days_since_precip": 0, "mean_temp_3d": 68, "snowfall_3d": 0.0, "freeze_thaw_3d": 0},
    {"scenario": "hero_dirt_target", "precip_1d": 0.00, "precip_3d": 0.18, "precip_7d": 0.30, "days_since_precip": 1, "mean_temp_3d": 66, "snowfall_3d": 0.0, "freeze_thaw_3d": 0},
    {"scenario": "damp", "precip_1d": 0.08, "precip_3d": 0.35, "precip_7d": 0.55, "days_since_precip": 0, "mean_temp_3d": 62, "snowfall_3d": 0.0, "freeze_thaw_3d": 0},
    {"scenario": "wet", "precip_1d": 0.25, "precip_3d": 0.70, "precip_7d": 1.00, "days_since_precip": 0, "mean_temp_3d": 58, "snowfall_3d": 0.0, "freeze_thaw_3d": 0},
    {"scenario": "very_wet", "precip_1d": 0.60, "precip_3d": 1.20, "precip_7d": 1.80, "days_since_precip": 0, "mean_temp_3d": 55, "snowfall_3d": 0.0, "freeze_thaw_3d": 0},
    {"scenario": "saturated", "precip_1d": 1.00, "precip_3d": 2.00, "precip_7d": 3.00, "days_since_precip": 0, "mean_temp_3d": 52, "snowfall_3d": 0.0, "freeze_thaw_3d": 0},
    {"scenario": "freeze_thaw", "precip_1d": 0.02, "precip_3d": 0.15, "precip_7d": 0.25, "days_since_precip": 0, "mean_temp_3d": 36, "snowfall_3d": 0.0, "freeze_thaw_3d": 3},
    {"scenario": "recent_snow", "precip_1d": 0.10, "precip_3d": 0.40, "precip_7d": 0.60, "days_since_precip": 0, "mean_temp_3d": 31, "snowfall_3d": 4.0, "freeze_thaw_3d": 1},
]

def build_test_rows():
    rows = []
    for terrain_name, terrain in TERRAIN_PROFILES.items():
        for scenario in SCENARIOS:
            rows.append({"terrain_profile": terrain_name, **scenario, **terrain})
    return pd.DataFrame(rows)

def main():
    output_path = PROJECT_ROOT / "data" / "processed" / "condition_model_v3_2_validation.csv"

    print("Building synthetic validation scenarios...")
    tests = build_test_rows()
    print(f"Test rows: {len(tests)}")

    print()
    print("Running v3.2...")
    predictions = tests.apply(predict_condition, axis=1)
    results = pd.concat(
        [tests.reset_index(drop=True), predictions.reset_index(drop=True)],
        axis=1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print()
    print("=" * 72)
    print("V3.2 MODEL STRESS TEST COMPLETE")
    print("=" * 72)
    print(f"Scenarios tested: {len(SCENARIOS)}")
    print(f"Terrain profiles: {len(TERRAIN_PROFILES)}")
    print(f"Total predictions: {len(results)}")
    print()
    print(f"Saved to:\n  {output_path}")
    print()
    print("Results:")
    print()

    display_columns = [
        "terrain_profile", "scenario", "precip_1d", "precip_3d",
        "precip_7d", "days_since_precip", "snowfall_3d",
        "freeze_thaw_3d", "estimated_moisture", "surface_state",
        "rideability_score", "rideability", "reason",
    ]
    print(results[display_columns].to_string(index=False))

    print()
    print("Domain checks:")
    failures = []

    for terrain_name in TERRAIN_PROFILES:
        subset = results[
            results["terrain_profile"] == terrain_name
        ].set_index("scenario")

        bone_dry = subset.loc["bone_dry_10_days", "rideability_score"]
        hero = subset.loc["hero_dirt_target", "rideability_score"]
        wet = subset.loc["wet", "rideability_score"]
        very_wet = subset.loc["very_wet", "rideability_score"]
        saturated = subset.loc["saturated", "rideability_score"]
        snow = subset.loc["recent_snow", "rideability_score"]

        if not hero > bone_dry:
            failures.append(
                f"{terrain_name}: hero dirt does not score above bone dry"
            )

        if not bone_dry > wet:
            failures.append(
                f"{terrain_name}: bone dry is not more rideable than wet"
            )

        # Equality is valid after the bounded score reaches its zero floor.
        if not (wet >= very_wet >= saturated):
            failures.append(
                f"{terrain_name}: wetness penalty is not monotonic"
            )

        # Before the zero floor, wetter conditions should still score worse.
        if not wet > very_wet:
            failures.append(
                f"{terrain_name}: very wet conditions are not penalized more than wet conditions"
            )

        if snow >= bone_dry:
            failures.append(
                f"{terrain_name}: snow is not penalized enough"
            )

    if failures:
        print(f"FAILED: {len(failures)} checks")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("PASS: all domain constraints satisfied.")

    print()
    print("Score matrix:")
    print()

    score_matrix = (
        results
        .pivot(
            index="scenario",
            columns="terrain_profile",
            values="rideability_score",
        )
        .reindex([scenario["scenario"] for scenario in SCENARIOS])
    )
    print(score_matrix.to_string())

if __name__ == "__main__":
    main()