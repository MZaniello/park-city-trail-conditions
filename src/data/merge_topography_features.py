from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[2]

    modeling_path = (
        project_root
        / "data"
        / "processed"
        / "trail_modeling_dataset.csv"
    )

    topography_path = (
        project_root
        / "data"
        / "processed"
        / "trail_topography_features.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "trail_modeling_dataset_v2.csv"
    )

    print("Loading modeling dataset...")

    dataset = pd.read_csv(
        modeling_path,
        parse_dates=["date"],
    )

    print(f"Rows before merge: {len(dataset):,}")

    print("Loading topography features...")

    topography = pd.read_csv(topography_path)

    print(
        f"Topography trails: "
        f"{topography['trail_name'].nunique()}"
    )

    # We don't need sampled_length_miles here because
    # distance already exists in the terrain dataset.
    features_to_add = topography[
        [
            "trail_name",
            "mean_slope_degrees",
            "median_slope_degrees",
            "north_facing_pct",
            "east_facing_pct",
            "south_facing_pct",
            "west_facing_pct",
            "topography_sample_count",
        ]
    ]

    print("Merging topography into modeling dataset...")

    merged = dataset.merge(
        features_to_add,
        on="trail_name",
        how="left",
        validate="many_to_one",
    )

    missing_topography = (
        merged["mean_slope_degrees"]
        .isna()
        .sum()
    )

    duplicate_rows = merged.duplicated(
        subset=[
            "date",
            "trail_name",
        ]
    ).sum()

    merged.to_csv(
        output_path,
        index=False,
    )

    print("\nMerge complete!")
    print(f"Rows: {len(merged):,}")
    print(f"Columns: {len(merged.columns)}")
    print(
        f"Trails: "
        f"{merged['trail_name'].nunique()}"
    )
    print(
        f"Duplicate trail/date rows: "
        f"{duplicate_rows:,}"
    )
    print(
        f"Missing topography rows: "
        f"{missing_topography:,}"
    )

    print(f"\nSaved to: {output_path}")

    print("\nExample topography features:")

    preview = (
        merged[
            [
                "trail_name",
                "mean_slope_degrees",
                "north_facing_pct",
                "east_facing_pct",
                "south_facing_pct",
                "west_facing_pct",
            ]
        ]
        .drop_duplicates("trail_name")
        .sort_values("trail_name")
    )

    print(
        preview.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()