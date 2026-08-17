from pathlib import Path

import osmnx as ox
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    graph_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_bike_network.graphml"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_route_segments.csv"
    )

    print("Loading Park City bike network...")

    graph = ox.load_graphml(graph_path)
    _, edges = ox.graph_to_gdfs(graph)

    useful_columns = [
        "osmid",
        "name",
        "highway",
        "length",
        "surface",
        "tracktype",
        "smoothness",
        "bicycle",
        "access",
        "mtb:scale",
        "mtb:scale:uphill",
        "width",
        "geometry",
    ]

    # Some OpenStreetMap fields may not exist in this download.
    available_columns = [
        column for column in useful_columns if column in edges.columns
    ]

    trail_data = edges[available_columns].copy()

    # Geometry cannot be stored cleanly in an ordinary CSV, so convert it to text.
    if "geometry" in trail_data.columns:
        trail_data["geometry"] = trail_data["geometry"].astype(str)

    trail_data = trail_data.reset_index()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trail_data.to_csv(output_path, index=False)

    print(f"Saved {len(trail_data):,} route segments.")
    print(f"Saved to: {output_path}")
    print()
    print("Columns found:")
    print(list(trail_data.columns))

    if "highway" in trail_data.columns:
        print()
        print("Most common route types:")
        print(trail_data["highway"].astype(str).value_counts().head(15))

    if "name" in trail_data.columns:
        named_count = trail_data["name"].notna().sum()
        print()
        print(f"Named segments: {named_count:,}")


if __name__ == "__main__":
    main()