from pathlib import Path

import osmnx as ox


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_bike_network.graphml"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_bike_network_elevation.graphml"
    )

    print("Loading Park City bike network...")

    graph = ox.load_graphml(input_path)

    print("Requesting elevation values for network nodes...")

    graph = ox.elevation.add_node_elevations_google(
        graph,
        api_key=None,
        batch_size=100,
        pause=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ox.save_graphml(
        graph,
        filepath=output_path,
    )

    elevations = [
        data.get("elevation")
        for _, data in graph.nodes(data=True)
        if data.get("elevation") is not None
    ]

    print(f"Nodes with elevation: {len(elevations):,}")
    print(f"Saved elevation network to: {output_path}")


if __name__ == "__main__":
    main()