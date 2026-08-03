from pathlib import Path

import pandas as pd


EXPECTED_REMOVED_TRAILS = {
    "Deer Valley Service",
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    catalog_path = (
        project_root
        / "data"
        / "processed"
        / "clean_trail_catalog.csv"
    )

    catalog = pd.read_csv(catalog_path)

    print(f"Rows in clean catalog: {len(catalog):,}")
    print(f"Unique trail names: {catalog['trail_name'].nunique():,}")

    duplicate_names = catalog[
        catalog["trail_name"].duplicated(keep=False)
    ]

    if duplicate_names.empty:
        print("No duplicate trail names found.")
    else:
        print("\nDuplicate trail names found:")
        print(duplicate_names["trail_name"].to_string(index=False))

    accidentally_kept = catalog[
        catalog["trail_name"].isin(EXPECTED_REMOVED_TRAILS)
    ]

    if accidentally_kept.empty:
        print("Known service routes were successfully excluded.")
    else:
        print("\nWarning: excluded routes are still present:")
        print(accidentally_kept["trail_name"].to_string(index=False))

    missing_lengths = catalog["total_length_miles"].isna().sum()
    nonpositive_lengths = (
        catalog["total_length_miles"] <= 0
    ).sum()

    print(f"Missing trail lengths: {missing_lengths}")
    print(f"Nonpositive trail lengths: {nonpositive_lengths}")

    print("\nClean catalog preview:")
    print(
        catalog[
            [
                "trail_name",
                "total_length_miles",
                "segment_count",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()