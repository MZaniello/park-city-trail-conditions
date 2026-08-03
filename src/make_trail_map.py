from pathlib import Path

import folium
import osmnx as ox


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    graph_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_bike_network.graphml"
    )

    output_path = (
        project_root
        / "outputs"
        / "maps"
        / "park_city_bike_network.html"
    )

    print("Loading Park City bike network...")

    graph = ox.load_graphml(graph_path)

    nodes, edges = ox.graph_to_gdfs(graph)

    center_lat = nodes.geometry.y.mean()
    center_lon = nodes.geometry.x.mean()

    trail_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    for _, edge in edges.iterrows():
        geometry = edge.geometry

        coordinates = [
            [latitude, longitude]
            for longitude, latitude in geometry.coords
        ]

        folium.PolyLine(
            coordinates,
            weight=2,
            opacity=0.7,
        ).add_to(trail_map)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    trail_map.save(output_path)

    print(f"Mapped {len(edges):,} route segments.")
    print(f"Saved map to: {output_path}")


if __name__ == "__main__":
    main()