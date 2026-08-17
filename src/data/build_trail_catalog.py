from pathlib import Path
from typing import Any

import numpy as np
import osmnx as ox
import pandas as pd


ALLOWED_HIGHWAY_TYPES = {"path", "track"}


def normalize_osm_value(value: Any) -> list[str]:
    """
    Convert an OpenStreetMap attribute into a clean list of strings.

    OSMnx may return values as strings, lists, tuples, or NumPy arrays.
    """
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    text = str(value).strip()

    if text.lower() in {"", "none", "nan"}:
        return []

    return [text]


def contains_allowed_type(value: Any) -> bool:
    """Return True if the highway value includes path or track."""
    highway_types = {
        item.lower()
        for item in normalize_osm_value(value)
    }

    return bool(highway_types & ALLOWED_HIGHWAY_TYPES)


def clean_trail_name(value: Any) -> str | None:
    """Convert an OpenStreetMap trail name into a clean string."""
    names = normalize_osm_value(value)

    names = [
        name
        for name in names
        if name.lower() not in {"unnamed", "unknown"}
    ]

    if not names:
        return None

    return " / ".join(names)


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    graph_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_bike_network.graphml"
    )

    segment_output_path = (
        project_root
        / "data"
        / "processed"
        / "named_trail_segments.csv"
    )

    catalog_output_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_trail_catalog.csv"
    )

    print("Loading Park City bicycle network...")

    graph = ox.load_graphml(graph_path)

    # Convert the directed bicycle network to an undirected network so
    # the same physical segment is not counted once in each direction.
    undirected_graph = ox.convert.to_undirected(graph)

    _, edges = ox.graph_to_gdfs(undirected_graph)
    trails = edges.reset_index().copy()

    print(f"Starting route segments: {len(trails):,}")

    trails["trail_name"] = trails["name"].apply(clean_trail_name)

    trails = trails[
        trails["trail_name"].notna()
        & trails["highway"].apply(contains_allowed_type)
    ].copy()

    trails["length_meters"] = pd.to_numeric(
        trails["length"],
        errors="coerce",
    )

    trails = trails.dropna(subset=["length_meters"])

    trails["length_miles"] = (
        trails["length_meters"] / 1609.344
    )

    segment_columns = [
        "u",
        "v",
        "key",
        "osmid",
        "trail_name",
        "highway",
        "length_meters",
        "length_miles",
    ]

    available_segment_columns = [
        column
        for column in segment_columns
        if column in trails.columns
    ]

    segment_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trails[available_segment_columns].to_csv(
        segment_output_path,
        index=False,
    )

    catalog = (
        trails.groupby(
            "trail_name",
            as_index=False,
        )
        .agg(
            segment_count=("trail_name", "size"),
            total_length_meters=("length_meters", "sum"),
            total_length_miles=("length_miles", "sum"),
        )
        .sort_values(
            by="total_length_miles",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    catalog["total_length_meters"] = (
        catalog["total_length_meters"].round(1)
    )

    catalog["total_length_miles"] = (
        catalog["total_length_miles"].round(2)
    )

    catalog.to_csv(
        catalog_output_path,
        index=False,
    )

    print(
        f"Named path/track segments retained: "
        f"{len(trails):,}"
    )
    print(
        f"Unique trail names found: "
        f"{len(catalog):,}"
    )
    print(
        f"Saved segment data to: "
        f"{segment_output_path}"
    )
    print(
        f"Saved trail catalog to: "
        f"{catalog_output_path}"
    )

    print(
        "\nTwenty longest named trails "
        "in this first-pass catalog:"
    )

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