from pathlib import Path
from typing import Any

import folium
import numpy as np
import osmnx as ox


ALLOWED_HIGHWAY_TYPES = {"path", "track"}


def normalize_osm_value(value: Any) -> list[str]:
    """Convert an OpenStreetMap value into a list of clean strings."""
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
    """Return True when a highway value includes path or track."""
    highway_types = {
        item.lower()
        for item in normalize_osm_value(value)
    }

    return bool(highway_types & ALLOWED_HIGHWAY_TYPES)


def clean_trail_name(value: Any) -> str | None:
    """Return a clean trail name."""
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

    map_output_path = (
        project_root
        / "outputs"
        / "maps"
        / "park_city_named_trails.html"
    )

    geojson_output_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_named_trail_segments.geojson"
    )

    print("Loading Park City bike network...")

    graph = ox.load_graphml(graph_path)
    graph = ox.convert.to_undirected(graph)

    nodes, edges = ox.graph_to_gdfs(graph)

    edges = edges.reset_index().copy()

    edges["trail_name"] = edges["name"].apply(clean_trail_name)

    trails = edges[
        edges["trail_name"].notna()
        & edges["highway"].apply(contains_allowed_type)
    ].copy()

    trails["length_miles"] = trails["length"] / 1609.344

    print(f"Named trail segments to map: {len(trails):,}")

    center_latitude = nodes.geometry.y.mean()
    center_longitude = nodes.geometry.x.mean()

    trail_map = folium.Map(
        location=[center_latitude, center_longitude],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    for _, trail in trails.iterrows():
        geometry = trail.geometry

        if geometry is None:
            continue

        coordinates = [
            [latitude, longitude]
            for longitude, latitude in geometry.coords
        ]

        highway_text = ", ".join(
            normalize_osm_value(trail["highway"])
        )

        popup_text = f"""
        <strong>{trail["trail_name"]}</strong><br>
        Type: {highway_text}<br>
        Segment length: {trail["length_miles"]:.2f} miles
        """

        folium.PolyLine(
            locations=coordinates,
            weight=4,
            opacity=0.8,
            tooltip=trail["trail_name"],
            popup=folium.Popup(
                popup_text,
                max_width=300,
            ),
        ).add_to(trail_map)

    map_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    geojson_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trail_map.save(map_output_path)

    trails.to_file(
        geojson_output_path,
        driver="GeoJSON",
    )

    print(f"Saved interactive map to: {map_output_path}")
    print(f"Saved trail GeoJSON to: {geojson_output_path}")


if __name__ == "__main__":
    main()