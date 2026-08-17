from pathlib import Path

import geopandas as gpd
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    geometry_path = (
        project_root
        / "data"
        / "processed"
        / "park_city_named_trail_segments.geojson"
    )

    catalog_path = (
        project_root
        / "data"
        / "processed"
        / "clean_trail_catalog.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "trail_locations.csv"
    )

    print("Loading trail geometries...")

    trails = gpd.read_file(geometry_path)
    catalog = pd.read_csv(catalog_path)

    approved_names = set(catalog["trail_name"])

    trails = trails[
        trails["trail_name"].isin(approved_names)
    ].copy()

    print(f"Approved trails found: {trails['trail_name'].nunique()}")

    # ---------------------------------------------------------
    # PROJECT GEOMETRIES
    #
    # We calculate representative points in a projected CRS
    # rather than directly in latitude/longitude.
    # ---------------------------------------------------------

    projected_crs = trails.estimate_utm_crs()

    trails_projected = trails.to_crs(projected_crs)

    locations = []

    for trail_name, group in trails_projected.groupby("trail_name"):

        combined_geometry = group.geometry.union_all()

        # representative_point guarantees that the point lies
        # on/within the combined geometry.
        point = combined_geometry.representative_point()

        point_gdf = gpd.GeoDataFrame(
            {
                "trail_name": [trail_name],
            },
            geometry=[point],
            crs=projected_crs,
        )

        point_wgs84 = point_gdf.to_crs("EPSG:4326")

        longitude = point_wgs84.geometry.x.iloc[0]
        latitude = point_wgs84.geometry.y.iloc[0]

        locations.append(
            {
                "trail_name": trail_name,
                "latitude": latitude,
                "longitude": longitude,
            }
        )

    locations = pd.DataFrame(locations)

    # ---------------------------------------------------------
    # ADD TERRAIN INFORMATION
    # ---------------------------------------------------------

    terrain_path = (
        project_root
        / "data"
        / "processed"
        / "trail_terrain_summary.csv"
    )

    terrain = pd.read_csv(terrain_path)

    locations = locations.merge(
        terrain,
        on="trail_name",
        how="left",
    )

    locations = locations.sort_values(
        "trail_name"
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    locations.to_csv(
        output_path,
        index=False,
    )

    print(f"Trail locations created: {len(locations)}")
    print(f"Saved to: {output_path}")

    print("\nTrail locations:")
    print(
        locations[
            [
                "trail_name",
                "latitude",
                "longitude",
                "minimum_elevation_feet",
                "maximum_elevation_feet",
            ]
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


if __name__ == "__main__":
    main()
    