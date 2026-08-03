from pathlib import Path

import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    catalog_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_trail_catalog.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "trail_review_table.csv"
    )

    print("Loading the first-pass trail catalog...")

    catalog = pd.read_csv(catalog_path)

    # Add columns that we will manually review.
    catalog["category"] = ""
    catalog["keep"] = ""
    catalog["notes"] = ""

    # Put the longest trails first because they are generally
    # the most important ones to review.
    catalog = catalog.sort_values(
        by="total_length_miles",
        ascending=False,
    )

    catalog.to_csv(output_path, index=False)

    print(f"Created review table with {len(catalog):,} trail names.")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()