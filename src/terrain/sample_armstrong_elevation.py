from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union


TRAIL_NAME = "Armstrong"
SAMPLE_SPACING_METERS = 20


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    trails_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_named_trail_segments.geojson"
    )

    dem_path = (
        project_root
        / "data"
        / "raw"
        / "park_city_dem_10m.tif"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "armstrong_elevation_samples.csv"
    )

    print("Loading trail geometry...")
    trails = gpd.read_file(trails_path)

    armstrong = trails[
        trails["trail_name"].str.casefold() == TRAIL_NAME.casefold()
    ].copy()

    if armstrong.empty:
        raise ValueError(f"No segments found for {TRAIL_NAME!r}.")

    print(f"Armstrong segments found: {len(armstrong)}")
    print(f"Trail CRS: {armstrong.crs}")

    dem = rioxarray.open_rasterio(
        dem_path,
        masked=True,
    )

    try:
        dem_crs = dem.rio.crs
        print(f"DEM CRS: {dem_crs}")

        # Reproject the trail into the DEM's coordinate system.
        armstrong = armstrong.to_crs(dem_crs)

        # Combine the separate Armstrong segments.
        merged_geometry = linemerge(
            unary_union(armstrong.geometry)
        )

        if merged_geometry.geom_type == "MultiLineString":
            # Keep the longest connected piece for this first test.
            merged_geometry = max(
                merged_geometry.geoms,
                key=lambda geometry: geometry.length,
            )

        if not isinstance(merged_geometry, LineString):
            raise TypeError(
                "Armstrong could not be converted into one LineString."
            )

        trail_length_meters = merged_geometry.length
        print(f"Sampled line length: {trail_length_meters:.1f} meters")

        distances = np.arange(
            0,
            trail_length_meters + SAMPLE_SPACING_METERS,
            SAMPLE_SPACING_METERS,
        )

        distances = np.clip(
            distances,
            0,
            trail_length_meters,
        )

        # Remove a duplicated endpoint if clipping created one.
        distances = np.unique(distances)

        points = [
            merged_geometry.interpolate(distance)
            for distance in distances
        ]

        elevations = []

        for point in points:
            elevation = dem.sel(
                x=point.x,
                y=point.y,
                method="nearest",
            ).item()

            elevations.append(float(elevation))

        samples = pd.DataFrame(
            {
                "distance_meters": distances,
                "distance_miles": distances / 1609.344,
                "elevation_meters": elevations,
                "elevation_feet": np.array(elevations) * 3.28084,
            }
        )

        samples.to_csv(output_path, index=False)

        print(f"Elevation samples created: {len(samples):,}")
        print(
            f"Minimum sampled elevation: "
            f"{samples['elevation_feet'].min():.0f} feet"
        )
        print(
            f"Maximum sampled elevation: "
            f"{samples['elevation_feet'].max():.0f} feet"
        )
        print(f"Saved samples to: {output_path}")

        print("\nFirst ten samples:")
        print(samples.head(10).to_string(index=False))

    finally:
        dem.close()

    print("DEM closed successfully.")


if __name__ == "__main__":
    main()