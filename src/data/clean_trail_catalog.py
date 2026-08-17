from pathlib import Path

import pandas as pd


# Trails we have manually reviewed and want to keep.
KEEP_TRAILS = {
    "Mid-Mountain Trail",
    "9K Trail",
    "Mother Urban",
    "Rambler",
    "CMG",
    "Lost Prospector",
    "Big Easy",
    "Jenni's Trail upper",
    "Spiro Trail",
    "Ripple Trail",
    "Armstrong",
    "Flagstaff Loop",
    "Tidal Wave",
    "Cyn City",
    "Keystone Trail",
    "Round Valley Express",
    "Solamere",
    "Empire Link",
    "Apex",
}


# Trails or routes we have reviewed and want to remove.
REMOVE_TRAILS = {
    "Deer Valley Service",
}

INCOMPLETE_TRAILS = {
    "Armstrong",
}

def classify_trail(trail_name: str) -> tuple[str, str, str]:
    """
    Assign a category, keep decision, and explanation to one trail name.
    """

    if trail_name in KEEP_TRAILS:
        note = "Manually reviewed and confirmed"

        if trail_name in INCOMPLETE_TRAILS:
            note = "Confirmed trail, but OpenStreetMap geometry is incomplete"

        return (
        "mountain_bike_trail",
        "yes",
        note,
        )


    
    if trail_name in REMOVE_TRAILS:
        return (
            "service_route",
            "no",
            "Manually reviewed and excluded",
        )

    # Automatically flag obvious service routes.
    if "service" in trail_name.lower():
        return (
            "service_route",
            "no",
            "Name suggests a service or access route",
        )

    # Everything else remains uncertain until reviewed.
    return (
        "uncertain",
        "",
        "Needs review",
    )


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_trail_catalog.csv"
    )

    review_output_path = (
        project_root
        / "data"
        / "processed"
        / "trail_review_table.csv"
    )

    clean_output_path = (
        project_root
        / "data"
        / "processed"
        / "clean_trail_catalog.csv"
    )

    print("Loading first-pass trail catalog...")

    catalog = pd.read_csv(input_path)

    classifications = catalog["trail_name"].apply(classify_trail)

    catalog[
        ["category", "keep", "notes"]
    ] = pd.DataFrame(
        classifications.tolist(),
        index=catalog.index,
    )

    catalog = catalog.sort_values(
        by="total_length_miles",
        ascending=False,
    )

    # Save the complete review table.
    catalog.to_csv(
        review_output_path,
        index=False,
    )

    # Save only trails explicitly marked yes.
    clean_catalog = catalog[
        catalog["keep"].str.lower() == "yes"
    ].copy()

    clean_catalog.to_csv(
        clean_output_path,
        index=False,
    )

    print(f"Total trail names reviewed: {len(catalog):,}")
    print(f"Trails currently kept: {len(clean_catalog):,}")
    print(f"Saved review table to: {review_output_path}")
    print(f"Saved clean catalog to: {clean_output_path}")

    print("\nTrails currently included:")
    print(
        clean_catalog[
            [
                "trail_name",
                "total_length_miles",
                "category",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()