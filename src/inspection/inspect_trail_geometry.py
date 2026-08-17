from pathlib import Path

import folium
import geopandas as gpd


TRAIL_NAME = "Armstrong"


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    trail_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_named_trail_segments.geojson"
    )

    output_path = (
        project_root
        / "outputs"
        / "maps"
        / "armstrong_geometry_inspection.html"
    )

    print("Loading named trail segments...")

    trails = gpd.read_file(trail_path)

    armstrong = trails[
        trails["trail_name"].str.casefold() == TRAIL_NAME.casefold()
    ].copy()

    if armstrong.empty:
        raise ValueError(f"No geometry found for {TRAIL_NAME!r}.")

    print(f"Armstrong segments found: {len(armstrong)}")

    armstrong = armstrong.to_crs("EPSG:4326")

    min_lon, min_lat, max_lon, max_lat = armstrong.total_bounds

    center_lat = (min_lat + max_lat) / 2
    center_lon = (min_lon + max_lon) / 2

    trail_map = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    segment_colors = [
        "red",
        "blue",
        "green",
        "purple",
        "orange",
        "darkred",
        "cadetblue",
        "darkgreen",
    ]

    for segment_number, (_, row) in enumerate(
        armstrong.iterrows(),
        start=1,
    ):
        geometry = row.geometry

        if geometry is None:
            continue

        color = segment_colors[
            (segment_number - 1) % len(segment_colors)
        ]

        coordinates = [
            [latitude, longitude]
            for longitude, latitude in geometry.coords
        ]

        popup_text = f"""
        <strong>{TRAIL_NAME}</strong><br>
        Segment number: {segment_number}<br>
        OSM ID: {row.get("osmid", "unknown")}<br>
        Segment length: {row.get("length", 0):.1f} meters
        """

        folium.PolyLine(
            locations=coordinates,
            color=color,
            weight=7,
            opacity=0.9,
            tooltip=f"Segment {segment_number}",
            popup=folium.Popup(
                popup_text,
                max_width=300,
            ),
        ).add_to(trail_map)

        start_coordinate = coordinates[0]
        end_coordinate = coordinates[-1]

        folium.Marker(
            location=start_coordinate,
            tooltip=f"Segment {segment_number} start",
            icon=folium.Icon(
                color="green",
                icon="play",
            ),
        ).add_to(trail_map)

        folium.Marker(
            location=end_coordinate,
            tooltip=f"Segment {segment_number} end",
            icon=folium.Icon(
                color="red",
                icon="stop",
            ),
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

    print(f"Saved inspection map to: {output_path}")


if __name__ == "__main__":
    main()