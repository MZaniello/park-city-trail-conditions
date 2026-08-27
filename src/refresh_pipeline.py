from pathlib import Path
import subprocess
import sys
import time


# ============================================================
# PIPELINE RUNNER
# ============================================================

def run_step(project_root, script_path, label, step_number, total_steps):
    print()
    print("=" * 70)
    print(f"{step_number}/{total_steps} - {label}")
    print("=" * 70)

    full_path = project_root / script_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"Pipeline script not found: {full_path}"
        )

    start_time = time.time()

    result = subprocess.run(
        [
            sys.executable,
            str(full_path),
        ],
        cwd=project_root,
    )

    elapsed = time.time() - start_time

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped because this step failed: {label}"
        )

    print()
    print(f"Completed in {elapsed:.1f} seconds.")


# ============================================================
# MAIN
# ============================================================

def main():
    project_root = Path(__file__).resolve().parents[1]

    steps = [
        (
            Path("src/forecast/download_master_group_forecast.py"),
            "Download current weather + 7-day forecast",
        ),
        (
            Path("src/forecast/build_master_forecast_features.py"),
            "Build rolling forecast features for all trails",
        ),
        (
            Path("src/forecast/predict_master_forecast_v3_2.py"),
            "Generate v3.2 trail-condition predictions",
        ),
        (
            Path("src/observations/download_condition_reports.py"),
            "Download rider and staff condition reports",
        ),
        (
            Path("src/observations/build_labeled_dataset.py"),
            "Update labeled condition-report dataset",
        ),
    ]

    total_steps = len(steps)

    print()
    print("=" * 70)
    print("PARK CITY TRAIL CONDITIONS")
    print("PRODUCTION DATA REFRESH")
    print("=" * 70)
    print()
    print(f"Project root: {project_root}")
    print(f"Python: {sys.executable}")
    print()
    print(
        "This refresh updates forecast weather, trail-condition "
        "predictions, and rider-report data."
    )
    print()
    print(
        "Static trail geometry, terrain features, historical weather, "
        "and the master trail catalog are not rebuilt."
    )

    pipeline_start = time.time()

    for step_number, (script_path, label) in enumerate(
        steps,
        start=1,
    ):
        run_step(
            project_root,
            script_path,
            label,
            step_number,
            total_steps,
        )

    elapsed = time.time() - pipeline_start

    print()
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print()
    print(f"Total runtime: {elapsed:.1f} seconds.")
    print()
    print("Updated dashboard inputs:")
    print(
        "  data/raw/master_trail_forecast.csv"
    )
    print(
        "  data/processed/master_trail_forecast_features.csv"
    )
    print(
        "  data/processed/"
        "master_forecast_condition_predictions_v3_2.csv"
    )
    print()
    print("Condition-report data has also been refreshed.")


if __name__ == "__main__":
    main()