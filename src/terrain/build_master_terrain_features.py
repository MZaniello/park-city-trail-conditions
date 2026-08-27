from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from rasterio.io import MemoryFile
from rasterio.transform import rowcol
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
)
from shapely.geometry import (
    LineString,
    MultiLineString,
)


# ============================================================
# SETTINGS
# ============================================================

PROJECTED_CRS = "EPSG:26912"

SAMPLE_SPACING_METERS = 20.0

METERS_TO_FEET = 3.280839895
METERS_TO_MILES = 1 / 1609.344


# ============================================================
# GEOMETRY HELPERS
# ============================================================


def extract_lines(geometry):
    """
    Return individual LineStrings from a trail geometry.

    Handles LineString and MultiLineString objects.
    """

    if geometry is None:
        return []

    if geometry.is_empty:
        return []

    if isinstance(
        geometry,
        LineString,
    ):
        return [geometry]

    if isinstance(
        geometry,
        MultiLineString,
    ):
        return list(
            geometry.geoms
        )

    # GeometryCollection fallback.
    if hasattr(
        geometry,
        "geoms",
    ):

        lines = []

        for part in geometry.geoms:

            if isinstance(
                part,
                LineString,
            ):
                lines.append(
                    part
                )

            elif isinstance(
                part,
                MultiLineString,
            ):
                lines.extend(
                    list(
                        part.geoms
                    )
                )

        return lines

    return []


def sample_points_along_geometry(
    geometry,
    spacing_meters,
):
    """
    Generate points approximately every N meters along all
    LineString components of a trail.
    """

    lines = extract_lines(
        geometry
    )

    points = []

    total_length = 0.0

    for line in lines:

        line_length = float(
            line.length
        )

        if line_length <= 0:
            continue

        total_length += (
            line_length
        )

        sample_count = max(
            2,
            int(
                np.ceil(
                    line_length
                    / spacing_meters
                )
            )
            + 1,
        )

        distances = np.linspace(
            0,
            line_length,
            sample_count,
        )

        for distance in distances:

            point = line.interpolate(
                distance
            )

            points.append(
                point
            )

    return (
        points,
        total_length,
    )


# ============================================================
# DEM PROCESSING
# ============================================================


def reproject_dem_to_projected_crs(
    source,
):
    """
    Reproject the source DEM into UTM Zone 12N.

    The downloaded USGS DEM is geographic (EPSG:4269). Slope
    cannot be calculated correctly using latitude/longitude
    degrees, so we first create a projected raster in meters.
    """

    print(
        "Reprojecting DEM to projected CRS "
        f"{PROJECTED_CRS}..."
    )

    transform, width, height = (
        calculate_default_transform(
            source.crs,
            PROJECTED_CRS,
            source.width,
            source.height,
            *source.bounds,
        )
    )

    profile = (
        source.profile.copy()
    )

    profile.update(
        {
            "crs":
                PROJECTED_CRS,

            "transform":
                transform,

            "width":
                width,

            "height":
                height,

            "dtype":
                "float32",

            "count":
                1,

            "nodata":
                -9999.0,
        }
    )

    memory_file = (
        MemoryFile()
    )

    projected = (
        memory_file.open(
            **profile
        )
    )

    reproject(
        source=rasterio.band(
            source,
            1,
        ),
        destination=rasterio.band(
            projected,
            1,
        ),
        src_transform=source.transform,
        src_crs=source.crs,
        src_nodata=source.nodata,
        dst_transform=transform,
        dst_crs=PROJECTED_CRS,
        dst_nodata=-9999.0,
        resampling=Resampling.bilinear,
    )

    return (
        memory_file,
        projected,
    )


def calculate_slope_and_aspect(
    dem,
):
    """
    Calculate slope and aspect arrays from a projected DEM.

    Slope is returned in degrees.

    Aspect convention:
        0   = North
        90  = East
        180 = South
        270 = West
    """

    print(
        "Calculating slope and aspect rasters..."
    )

    elevation = dem.read(
        1
    ).astype(
        "float64"
    )

    nodata = dem.nodata

    if nodata is not None:

        invalid = np.isclose(
            elevation,
            nodata,
        )

    else:

        invalid = ~np.isfinite(
            elevation
        )

    elevation[
        invalid
    ] = np.nan

    x_resolution = abs(
        dem.transform.a
    )

    y_resolution = abs(
        dem.transform.e
    )

    # Fill holes only for numerical gradient calculation.
    #
    # The valid-data mask will still prevent these filled cells
    # from being sampled later.
    finite_values = elevation[
        np.isfinite(
            elevation
        )
    ]

    if finite_values.size == 0:

        raise RuntimeError(
            "Projected DEM contains no valid elevation values."
        )

    fill_value = float(
        np.nanmedian(
            finite_values
        )
    )

    elevation_filled = (
        np.where(
            np.isfinite(
                elevation
            ),
            elevation,
            fill_value,
        )
    )

    dz_dy, dz_dx = np.gradient(
        elevation_filled,
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

    # Aspect measured clockwise from north.
    aspect = np.degrees(
        np.arctan2(
            dz_dx,
            -dz_dy,
        )
    )

    aspect = (
        aspect
        + 360.0
    ) % 360.0

    slope_degrees[
        invalid
    ] = np.nan

    aspect[
        invalid
    ] = np.nan

    return (
        elevation,
        slope_degrees,
        aspect,
    )


# ============================================================
# SAMPLE HELPERS
# ============================================================


def sample_rasters(
    points,
    dem,
    elevation,
    slope,
    aspect,
):
    """
    Sample elevation, slope and aspect for a list of projected
    trail points.
    """

    elevations = []
    slopes = []
    aspects = []

    height, width = (
        elevation.shape
    )

    for point in points:

        try:

            row, col = rowcol(
                dem.transform,
                point.x,
                point.y,
            )

        except Exception:
            continue

        if (
            row < 0
            or row >= height
            or col < 0
            or col >= width
        ):
            continue

        elevation_value = (
            elevation[
                row,
                col,
            ]
        )

        slope_value = (
            slope[
                row,
                col,
            ]
        )

        aspect_value = (
            aspect[
                row,
                col,
            ]
        )

        if not (
            np.isfinite(
                elevation_value
            )
            and np.isfinite(
                slope_value
            )
            and np.isfinite(
                aspect_value
            )
        ):
            continue

        elevations.append(
            float(
                elevation_value
            )
        )

        slopes.append(
            float(
                slope_value
            )
        )

        aspects.append(
            float(
                aspect_value
            )
        )

    return (
        np.asarray(
            elevations
        ),
        np.asarray(
            slopes
        ),
        np.asarray(
            aspects
        ),
    )


def aspect_percentages(
    aspects,
):
    """
    Divide aspect into four broad directional groups.

    North:
        >= 315 or < 45

    East:
        45 to < 135

    South:
        135 to < 225

    West:
        225 to < 315
    """

    if len(
        aspects
    ) == 0:

        return {
            "north_facing_pct":
                np.nan,

            "east_facing_pct":
                np.nan,

            "south_facing_pct":
                np.nan,

            "west_facing_pct":
                np.nan,
        }

    north = (
        (aspects >= 315)
        | (aspects < 45)
    )

    east = (
        (aspects >= 45)
        & (aspects < 135)
    )

    south = (
        (aspects >= 135)
        & (aspects < 225)
    )

    west = (
        (aspects >= 225)
        & (aspects < 315)
    )

    total = len(
        aspects
    )

    return {
        "north_facing_pct":
            north.sum()
            / total
            * 100,

        "east_facing_pct":
            east.sum()
            / total
            * 100,

        "south_facing_pct":
            south.sum()
            / total
            * 100,

        "west_facing_pct":
            west.sum()
            / total
            * 100,
    }


# ============================================================
# MAIN
# ============================================================


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    trails_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_geometries.geojson"
    )

    dem_path = (
        project_root
        / "data"
        / "raw"
        / "expanded_park_city_dem_10m.tif"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_topography_features.csv"
    )

    # ---------------------------------------------------------
    # LOAD TRAILS
    # ---------------------------------------------------------

    print(
        "Loading master trail geometries..."
    )

    trails = gpd.read_file(
        trails_path
    )

    print(
        f"Trails found: "
        f"{len(trails):,}"
    )

    if trails.empty:

        raise RuntimeError(
            "No master trail geometries were found."
        )

    if (
        trails[
            "trail_name"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate trail names exist in "
            "master_trail_geometries.geojson."
        )

    # ---------------------------------------------------------
    # PROJECT TRAILS
    # ---------------------------------------------------------

    print(
        "Projecting trail geometry..."
    )

    trails_projected = (
        trails
        .to_crs(
            PROJECTED_CRS
        )
        .copy()
    )

    # ---------------------------------------------------------
    # LOAD DEM
    # ---------------------------------------------------------

    print(
        "Loading expanded DEM..."
    )

    with rasterio.open(
        dem_path
    ) as source_dem:

        print(
            f"Source DEM CRS: "
            f"{source_dem.crs}"
        )

        print(
            f"Source DEM size: "
            f"{source_dem.width:,} x "
            f"{source_dem.height:,}"
        )

        (
            memory_file,
            projected_dem,
        ) = reproject_dem_to_projected_crs(
            source_dem
        )

        try:

            print(
                f"Projected DEM size: "
                f"{projected_dem.width:,} x "
                f"{projected_dem.height:,}"
            )

            print(
                "Projected resolution: "
                f"{projected_dem.res}"
            )

            (
                elevation,
                slope,
                aspect,
            ) = calculate_slope_and_aspect(
                projected_dem
            )

            # -------------------------------------------------
            # ANALYZE TRAILS
            # -------------------------------------------------

            records = []

            total_trails = len(
                trails_projected
            )

            print()
            print(
                "Analyzing trails..."
            )

            for number, (
                _,
                trail,
            ) in enumerate(
                trails_projected.iterrows(),
                start=1,
            ):

                trail_name = (
                    trail[
                        "trail_name"
                    ]
                )

                if (
                    number == 1
                    or number % 25 == 0
                    or number == total_trails
                ):

                    print(
                        f"[{number}/{total_trails}] "
                        f"{trail_name}"
                    )

                (
                    sample_points,
                    sampled_length_meters,
                ) = sample_points_along_geometry(
                    trail.geometry,
                    SAMPLE_SPACING_METERS,
                )

                (
                    elevations,
                    slopes,
                    aspects,
                ) = sample_rasters(
                    sample_points,
                    projected_dem,
                    elevation,
                    slope,
                    aspect,
                )

                if len(
                    elevations
                ) == 0:

                    print(
                        f"  WARNING: no valid DEM "
                        f"samples for {trail_name}"
                    )

                    record = {
                        "trail_name":
                            trail_name,

                        "sampled_length_miles":
                            sampled_length_meters
                            * METERS_TO_MILES,

                        "topography_sample_count":
                            0,

                        "minimum_elevation_feet":
                            np.nan,

                        "maximum_elevation_feet":
                            np.nan,

                        "mean_elevation_feet":
                            np.nan,

                        "elevation_range_feet":
                            np.nan,

                        "mean_slope_degrees":
                            np.nan,

                        "median_slope_degrees":
                            np.nan,

                        "maximum_slope_degrees":
                            np.nan,

                        "north_facing_pct":
                            np.nan,

                        "east_facing_pct":
                            np.nan,

                        "south_facing_pct":
                            np.nan,

                        "west_facing_pct":
                            np.nan,
                    }

                else:

                    elevation_feet = (
                        elevations
                        * METERS_TO_FEET
                    )

                    aspect_stats = (
                        aspect_percentages(
                            aspects
                        )
                    )

                    record = {
                        "trail_name":
                            trail_name,

                        "sampled_length_miles":
                            sampled_length_meters
                            * METERS_TO_MILES,

                        "topography_sample_count":
                            len(
                                elevations
                            ),

                        "minimum_elevation_feet":
                            np.min(
                                elevation_feet
                            ),

                        "maximum_elevation_feet":
                            np.max(
                                elevation_feet
                            ),

                        "mean_elevation_feet":
                            np.mean(
                                elevation_feet
                            ),

                        "elevation_range_feet":
                            (
                                np.max(
                                    elevation_feet
                                )
                                - np.min(
                                    elevation_feet
                                )
                            ),

                        "mean_slope_degrees":
                            np.mean(
                                slopes
                            ),

                        "median_slope_degrees":
                            np.median(
                                slopes
                            ),

                        "maximum_slope_degrees":
                            np.max(
                                slopes
                            ),

                        **aspect_stats,
                    }

                # Preserve catalog metadata where available.
                for column in [
                    "final_area",
                    "latitude",
                    "longitude",
                    "distance_miles",
                    "source",
                ]:

                    if column in trail.index:

                        record[
                            column
                        ] = trail[
                            column
                        ]

                records.append(
                    record
                )

        finally:

            projected_dem.close()
            memory_file.close()

    # ---------------------------------------------------------
    # BUILD OUTPUT
    # ---------------------------------------------------------

    results = pd.DataFrame(
        records
    )

    results = results.sort_values(
        [
            "final_area",
            "trail_name",
        ]
        if "final_area"
        in results.columns
        else [
            "trail_name"
        ]
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    duplicate_names = (
        results[
            "trail_name"
        ]
        .duplicated()
        .sum()
    )

    missing_topography = (
        results[
            "mean_slope_degrees"
        ]
        .isna()
        .sum()
    )

    missing_elevation = (
        results[
            "minimum_elevation_feet"
        ]
        .isna()
        .sum()
    )

    aspect_sum = (
        results[
            [
                "north_facing_pct",
                "east_facing_pct",
                "south_facing_pct",
                "west_facing_pct",
            ]
        ]
        .sum(
            axis=1
        )
    )

    valid_aspect_rows = (
        results[
            "mean_slope_degrees"
        ]
        .notna()
    )

    bad_aspect_sums = (
        (
            aspect_sum[
                valid_aspect_rows
            ]
            - 100.0
        )
        .abs()
        > 0.1
    ).sum()

    # ---------------------------------------------------------
    # ROUND DISPLAY / OUTPUT VALUES
    # ---------------------------------------------------------

    round_columns = [
        "sampled_length_miles",
        "minimum_elevation_feet",
        "maximum_elevation_feet",
        "mean_elevation_feet",
        "elevation_range_feet",
        "mean_slope_degrees",
        "median_slope_degrees",
        "maximum_slope_degrees",
        "north_facing_pct",
        "east_facing_pct",
        "south_facing_pct",
        "west_facing_pct",
    ]

    for column in round_columns:

        if column in results.columns:

            results[
                column
            ] = (
                results[
                    column
                ]
                .round(1)
            )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MASTER TRAIL TERRAIN EXTRACTION COMPLETE"
    )
    print("=" * 72)

    print(
        f"Trails analyzed: "
        f"{len(results):,}"
    )

    print(
        f"Duplicate trail names: "
        f"{duplicate_names:,}"
    )

    print(
        f"Missing elevation rows: "
        f"{missing_elevation:,}"
    )

    print(
        f"Missing topography rows: "
        f"{missing_topography:,}"
    )

    print(
        f"Aspect percentage errors: "
        f"{bad_aspect_sums:,}"
    )

    print()
    print(
        f"Saved to:"
        f"\n  {output_path}"
    )

    # ---------------------------------------------------------
    # RANGE CHECKS
    # ---------------------------------------------------------

    valid = results[
        results[
            "minimum_elevation_feet"
        ]
        .notna()
    ]

    if not valid.empty:

        print()
        print(
            "Dataset terrain ranges:"
        )

        print(
            "  Elevation: "
            f"{valid['minimum_elevation_feet'].min():,.0f}"
            "–"
            f"{valid['maximum_elevation_feet'].max():,.0f} ft"
        )

        print(
            "  Mean slope: "
            f"{valid['mean_slope_degrees'].min():.1f}"
            "–"
            f"{valid['mean_slope_degrees'].max():.1f}°"
        )

    # ---------------------------------------------------------
    # EXAMPLE OUTPUT
    # ---------------------------------------------------------

    print()
    print(
        "Topography sample:"
    )

    preview_columns = [
        "trail_name",
        "final_area",
        "sampled_length_miles",
        "minimum_elevation_feet",
        "maximum_elevation_feet",
        "mean_slope_degrees",
        "north_facing_pct",
        "east_facing_pct",
        "south_facing_pct",
        "west_facing_pct",
    ]

    preview_columns = [
        column
        for column in preview_columns
        if column in results.columns
    ]

    print()

    print(
        results[
            preview_columns
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()