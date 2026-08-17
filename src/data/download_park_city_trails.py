from pathlib import Path

import osmnx as ox


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "raw" / "park_city_bike_network.graphml"

    print("Downloading bicycle-accessible routes around Park City...")

    graph = ox.graph_from_place(
        "Park City, Utah, USA",
        network_type="bike",
        simplify=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=output_path)

    print(f"Downloaded {len(graph.nodes):,} nodes.")
    print(f"Downloaded {len(graph.edges):,} route segments.")
    print(f"Saved data to: {output_path}")


if __name__ == "__main__":
    main()