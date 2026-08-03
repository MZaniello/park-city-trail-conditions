from pathlib import Path

import geopandas as gpd
import py3dep


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    trail_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_named_trail_segments.geojson"
    )

    output_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_dem_10m.tif"
    )

    print("Loading trail geometries...")

    trails = gpd.read_file(trail_path)

    # py3dep expects geographic coordinates in longitude/latitude.
    trails = trails.to_crs("EPSG:4326")

    min_lon, min_lat, max_lon, max_lat = trails.total_bounds

    # Add a small buffer around the mapped trail area.
    longitude_buffer = 0.01
    latitude_buffer = 0.01

    bounding_box = (
        min_lon - longitude_buffer,
        min_lat - latitude_buffer,
        max_lon + longitude_buffer,
        max_lat + latitude_buffer,
    )

    print("Downloading a 10-meter USGS 3DEP elevation model...")
    print(f"Bounding box: {bounding_box}")

    dem = py3dep.get_dem(
        bounding_box,
        resolution=10,
        crs="EPSG:4326",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dem.rio.to_raster(output_path)

    print(f"DEM dimensions: {dem.shape}")
    print(f"Saved elevation raster to: {output_path}")


if __name__ == "__main__":
    main()