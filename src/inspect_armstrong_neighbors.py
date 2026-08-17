from pathlib import Path
from typing import Any

import folium
import geopandas as gpd
import numpy as np
import osmnx as ox


TRAIL_NAME = "Armstrong"
SEARCH_BUFFER_METERS = 250


def normalize_value(value: Any) -> str:
    """Convert messy OpenStreetMap values into readable text."""
    if value is None:
        return "Unnamed"

    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (list, tuple, set)):
        return " / ".join(str(item) for item in value)

    text = str(value).strip()

    if text.lower() in {"", "none", "nan"}:
        return "Unnamed"

    return text


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    graph_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_bike_network.graphml"
    )

    named_trails_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_named_trail_segments.geojson"
    )

    output_path = (
        project_root
        / "outputs"
        / "maps"
        / "armstrong_neighbor_inspection.html"
    )

    print("Loading Armstrong geometry...")

    named_trails = gpd.read_file(named_trails_path)

    armstrong = named_trails[
        named_trails["trail_name"].str.casefold()
        == TRAIL_NAME.casefold()
    ].copy()

    if armstrong.empty:
        raise ValueError("No Armstrong geometry found.")

    print("Loading complete bicycle network...")

    graph = ox.load_graphml(graph_path)
    graph = ox.convert.to_undirected(graph)

    _, all_edges = ox.graph_to_gdfs(graph)

    # Use a projected coordinate system so the buffer is measured in meters.
    projected_crs = all_edges.estimate_utm_crs()

    all_edges_projected = all_edges.to_crs(projected_crs)
    armstrong_projected = armstrong.to_crs(projected_crs)

    search_area = armstrong_projected.geometry.union_all().buffer(
        SEARCH_BUFFER_METERS
    )

    nearby_edges = all_edges_projected[
        all_edges_projected.intersects(search_area)
    ].copy()

    print(f"Nearby route segments found: {len(nearby_edges):,}")

    nearby_edges["display_name"] = nearby_edges["name"].apply(
        normalize_value
    )

    nearby_edges["display_highway"] = nearby_edges["highway"].apply(
        normalize_value
    )

    print("\nNearby names:")
    print(
        nearby_edges["display_name"]
        .value_counts()
        .head(30)
        .to_string()
    )

    nearby_edges = nearby_edges.to_crs("EPSG:4326")
    armstrong = armstrong.to_crs("EPSG:4326")

    min_lon, min_lat, max_lon, max_lat = nearby_edges.total_bounds

    trail_map = folium.Map(
        location=[
            (min_lat + max_lat) / 2,
            (min_lon + max_lon) / 2,
        ],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # Draw every nearby segment.
    for _, row in nearby_edges.iterrows():
        geometry = row.geometry

        if geometry is None:
            continue

        coordinates = [
            [latitude, longitude]
            for longitude, latitude in geometry.coords
        ]

        popup_text = f"""
        <strong>{row["display_name"]}</strong><br>
        Highway type: {row["display_highway"]}<br>
        Length: {float(row.get("length", 0)):.1f} meters<br>
        OSM ID: {row.get("osmid", "unknown")}
        """

        folium.PolyLine(
            locations=coordinates,
            weight=3,
            opacity=0.6,
            tooltip=row["display_name"],
            popup=folium.Popup(
                popup_text,
                max_width=350,
            ),
        ).add_to(trail_map)

    # Draw the currently recognized Armstrong segments more prominently.
    for _, row in armstrong.iterrows():
        coordinates = [
            [latitude, longitude]
            for longitude, latitude in row.geometry.coords
        ]

        folium.PolyLine(
            locations=coordinates,
            weight=8,
            opacity=1.0,
            tooltip="Currently recognized Armstrong",
        ).add_to(trail_map)

    trail_map.fit_bounds(
        [
            [min_lat, min_lon],
            [max_lat, max_lon],
        ]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trail_map.save(output_path)

    print(f"\nSaved inspection map to: {output_path}")


if __name__ == "__main__":
    main()