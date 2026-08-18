from pathlib import Path
import subprocess
import sys


def run_step(project_root, script_path, label):
    print()
    print("=" * 60)
    print(label)
    print("=" * 60)

    full_path = project_root / script_path

    result = subprocess.run(
        [
            sys.executable,
            str(full_path),
        ],
        cwd=project_root,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pipeline stopped because this step failed: {label}"
        )


def main():
    project_root = Path(__file__).resolve().parents[1]

    steps = [
        (
            Path("src/weather/download_trail_weather.py"),
            "1/5 — Download trail-specific weather",
        ),
        (
            Path("src/weather/build_trail_weather_features.py"),
            "2/5 — Build rolling weather features",
        ),
        (
            Path("src/data/merge_topography_features.py"),
            "3/5 — Merge topography features",
        ),
        (
            Path("src/observations/download_condition_reports.py"),
            "4/5 — Download condition reports",
        ),
        (
            Path("src/observations/build_labeled_dataset.py"),
            "5/5 — Build labeled modeling dataset",
        ),
    ]

    print("Refreshing Park City trail data pipeline...")

    for script_path, label in steps:
        run_step(
            project_root,
            script_path,
            label,
        )

    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(
        "Weather, terrain features, condition reports, "
        "and labeled data are now up to date."
    )


if __name__ == "__main__":
    main()