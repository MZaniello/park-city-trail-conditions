from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
from shapely.geometry import LineString
from shapely.ops import linemerge, unary_union


SAMPLE_SPACING_METERS = 20
SMOOTHING_WINDOW = 5


def merge_trail_geometry(geometry_series):
    merged = linemerge(unary_union(geometry_series))

    if merged.geom_type == "MultiLineString":
        merged = max(
            merged.geoms,
            key=lambda geometry: geometry.length,
        )

    if not isinstance(merged, LineString):
        return None

    return merged


def sample_trail_elevations(line, dem):
    distances = np.arange(
        0,
        line.length + SAMPLE_SPACING_METERS,
        SAMPLE_SPACING_METERS,
    )

    distances = np.unique(
        np.clip(distances, 0, line.length)
    )

    elevations = []

    for distance in distances:
        point = line.interpolate(distance)

        elevation = dem.sel(
            x=point.x,
            y=point.y,
            method="nearest",
        ).item()

        elevations.append(float(elevation))

    return distances, np.array(elevations)


def summarize_trail(
    trail_name,
    distances,
    elevations,
):
    smoothed = (
        pd.Series(elevations)
        .rolling(
            window=SMOOTHING_WINDOW,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )

    elevation_changes = np.diff(smoothed)

    gain_meters = elevation_changes[
        elevation_changes > 0
    ].sum()

    loss_meters = -elevation_changes[
        elevation_changes < 0
    ].sum()

    total_distance_meters = distances[-1]

    return {
        "trail_name": trail_name,
        "distance_miles": total_distance_meters / 1609.344,
        "minimum_elevation_feet": elevations.min() * 3.28084,
        "maximum_elevation_feet": elevations.max() * 3.28084,
        "elevation_gain_feet": gain_meters * 3.28084,
        "elevation_loss_feet": loss_meters * 3.28084,
        "elevation_range_feet": (
            elevations.max() - elevations.min()
        ) * 3.28084,
        "sample_count": len(elevations),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    trail_segments_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_named_trail_segments.geojson"
    )

    clean_catalog_path = (
        project_root
        / "data"
        / "processed"
        / "clean_trail_catalog.csv"
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
        / "trail_terrain_summary.csv"
    )

    print("Loading approved trail catalog...")
    catalog = pd.read_csv(clean_catalog_path)

    approved_names = set(catalog["trail_name"])

    print("Loading trail geometries...")
    trails = gpd.read_file(trail_segments_path)

    trails = trails[
        trails["trail_name"].isin(approved_names)
    ].copy()

    dem = rioxarray.open_rasterio(
        dem_path,
        masked=True,
    )

    summaries = []

    try:
        trails = trails.to_crs(dem.rio.crs)

        for trail_name in sorted(approved_names):
            print(f"Analyzing {trail_name}...")

            trail_parts = trails[
                trails["trail_name"] == trail_name
            ]

            if trail_parts.empty:
                print("  Skipped: no matching geometry.")
                continue

            line = merge_trail_geometry(
                trail_parts.geometry
            )

            if line is None:
                print("  Skipped: could not form a line.")
                continue

            distances, elevations = sample_trail_elevations(
                line,
                dem,
            )

            if len(elevations) < 2 or np.isnan(elevations).any():
                print("  Skipped: invalid elevation values.")
                continue

            summaries.append(
                summarize_trail(
                    trail_name,
                    distances,
                    elevations,
                )
            )

    finally:
        dem.close()

    summary_table = pd.DataFrame(summaries)

    numeric_columns = [
        "distance_miles",
        "minimum_elevation_feet",
        "maximum_elevation_feet",
        "elevation_gain_feet",
        "elevation_loss_feet",
        "elevation_range_feet",
    ]

    summary_table[numeric_columns] = (
        summary_table[numeric_columns].round(1)
    )

    summary_table = summary_table.sort_values(
        "elevation_gain_feet",
        ascending=False,
    )

    summary_table.to_csv(output_path, index=False)

    print(f"\nTrails successfully analyzed: {len(summary_table)}")
    print(f"Saved results to: {output_path}")

    print("\nTrail terrain summary:")
    print(
        summary_table[
            [
                "trail_name",
                "distance_miles",
                "elevation_gain_feet",
                "minimum_elevation_feet",
                "maximum_elevation_feet",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()