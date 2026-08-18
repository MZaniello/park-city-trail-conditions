from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
from rasterio.transform import rowcol


SAMPLE_SPACING_METERS = 20


def classify_aspect(aspect):
    """Convert an aspect angle into a cardinal direction."""

    if aspect >= 315 or aspect < 45:
        return "north"

    if aspect < 135:
        return "east"

    if aspect < 225:
        return "south"

    return "west"


def main():
    project_root = Path(__file__).resolve().parents[2]

    trail_path = (
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
        / "trail_topography_features.csv"
    )

    print("Loading trails...")

    trails = gpd.read_file(trail_path)
    catalog = pd.read_csv(catalog_path)

    approved_names = set(catalog["trail_name"])

    trails = trails[
        trails["trail_name"].isin(approved_names)
    ].copy()

    print(
        f"Trails found: "
        f"{trails['trail_name'].nunique()}"
    )

    print("Loading DEM...")

    dem = rioxarray.open_rasterio(
        dem_path,
        masked=True,
    ).squeeze()

    try:
        # Put trails into the same coordinate reference system
        # as the elevation raster.
        trails = trails.to_crs(dem.rio.crs)

        print("Calculating slope and aspect rasters...")

        elevation = dem.values

        x_resolution = abs(
            dem.rio.resolution()[0]
        )

        y_resolution = abs(
            dem.rio.resolution()[1]
        )

        # -----------------------------------------------------
        # SLOPE
        # -----------------------------------------------------

        dz_dy, dz_dx = np.gradient(
            elevation,
            y_resolution,
            x_resolution,
        )

        slope_radians = np.arctan(
            np.sqrt(
                dz_dx ** 2
                + dz_dy ** 2
            )
        )

        slope_degrees = np.degrees(
            slope_radians
        )

        # -----------------------------------------------------
        # ASPECT
        #
        # 0°   = north
        # 90°  = east
        # 180° = south
        # 270° = west
        # -----------------------------------------------------

        aspect_degrees = (
            np.degrees(
                np.arctan2(
                    dz_dx,
                    -dz_dy,
                )
            )
            + 360
        ) % 360

        # -----------------------------------------------------
        # SAMPLE EVERY APPROVED TRAIL
        # -----------------------------------------------------

        results = []

        for trail_name in sorted(approved_names):

            print(
                f"Analyzing {trail_name}..."
            )

            trail_parts = trails[
                trails["trail_name"]
                == trail_name
            ]

            if trail_parts.empty:
                print(
                    "  Skipped: no geometry."
                )
                continue

            slope_samples = []
            aspect_samples = []

            total_sampled_length = 0.0

            # -------------------------------------------------
            # IMPORTANT:
            #
            # Analyze ALL mapped pieces of a trail rather than
            # keeping only the longest connected geometry.
            # -------------------------------------------------

            for geometry in trail_parts.geometry:

                if geometry is None:
                    continue

                if geometry.geom_type == "LineString":
                    lines = [geometry]

                elif geometry.geom_type == "MultiLineString":
                    lines = list(
                        geometry.geoms
                    )

                else:
                    continue

                for line in lines:

                    total_sampled_length += (
                        line.length
                    )

                    distances = np.arange(
                        0,
                        line.length,
                        SAMPLE_SPACING_METERS,
                    )

                    if len(distances) == 0:
                        distances = np.array(
                            [0]
                        )

                    for distance in distances:

                        point = line.interpolate(
                            distance
                        )

                        row, col = rowcol(
                            dem.rio.transform(),
                            point.x,
                            point.y,
                        )

                        if (
                            row < 0
                            or col < 0
                            or row
                            >= slope_degrees.shape[0]
                            or col
                            >= slope_degrees.shape[1]
                        ):
                            continue

                        slope = (
                            slope_degrees[
                                row,
                                col,
                            ]
                        )

                        aspect = (
                            aspect_degrees[
                                row,
                                col,
                            ]
                        )

                        if (
                            np.isnan(slope)
                            or np.isnan(aspect)
                        ):
                            continue

                        slope_samples.append(
                            float(slope)
                        )

                        aspect_samples.append(
                            float(aspect)
                        )

            if not slope_samples:
                print(
                    "  Skipped: no valid "
                    "raster samples."
                )
                continue

            # -------------------------------------------------
            # CARDINAL ASPECT DISTRIBUTION
            # -------------------------------------------------

            directions = [
                classify_aspect(aspect)
                for aspect
                in aspect_samples
            ]

            sample_count = len(
                directions
            )

            north_pct = (
                directions.count("north")
                / sample_count
                * 100
            )

            east_pct = (
                directions.count("east")
                / sample_count
                * 100
            )

            south_pct = (
                directions.count("south")
                / sample_count
                * 100
            )

            west_pct = (
                directions.count("west")
                / sample_count
                * 100
            )

            # -------------------------------------------------
            # SAVE RESULTS FOR THIS TRAIL
            # -------------------------------------------------

            results.append(
                {
                    "trail_name": trail_name,

                    "sampled_length_miles":
                        total_sampled_length
                        / 1609.344,

                    "mean_slope_degrees":
                        np.mean(
                            slope_samples
                        ),

                    "median_slope_degrees":
                        np.median(
                            slope_samples
                        ),

                    "north_facing_pct":
                        north_pct,

                    "east_facing_pct":
                        east_pct,

                    "south_facing_pct":
                        south_pct,

                    "west_facing_pct":
                        west_pct,

                    "topography_sample_count":
                        sample_count,
                }
            )

    finally:
        dem.close()

    # ---------------------------------------------------------
    # BUILD FINAL TABLE
    # ---------------------------------------------------------

    features = pd.DataFrame(
        results
    )

    numeric_columns = [
        "sampled_length_miles",
        "mean_slope_degrees",
        "median_slope_degrees",
        "north_facing_pct",
        "east_facing_pct",
        "south_facing_pct",
        "west_facing_pct",
    ]

    features[numeric_columns] = (
        features[numeric_columns]
        .round(1)
    )

    features = features.sort_values(
        "trail_name"
    ).reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    features.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    print(
        "\nTerrain feature extraction complete!"
    )

    print(
        f"Trails analyzed: "
        f"{len(features)}"
    )

    print(
        f"Saved to: "
        f"{output_path}"
    )

    print(
        "\nTopography summary:"
    )

    print(
        features[
            [
                "trail_name",
                "sampled_length_miles",
                "mean_slope_degrees",
                "north_facing_pct",
                "east_facing_pct",
                "south_facing_pct",
                "west_facing_pct",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()