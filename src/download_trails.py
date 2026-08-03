from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "processed" / "setup_check.csv"

    sample_data = pd.DataFrame(
        {
            "trail_name": ["Armstrong", "Spiro", "Mid Mountain"],
            "status": ["planned", "planned", "planned"],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_data.to_csv(output_path, index=False)

    print(f"Created: {output_path}")
    print(sample_data)


if __name__ == "__main__":
    main()