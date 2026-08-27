from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

# Approximate horizontal grouping size.
#
# ~0.025 degrees latitude is roughly 1.7 miles / 2.8 km.
# Longitude spacing varies slightly with latitude but is close
# enough for this first-pass weather grouping.
LAT_GRID_SIZE = 0.025
LON_GRID_SIZE = 0.030

# Elevation matters strongly for temperature / snow.
# Keep trails in separate ~500 ft elevation bands.
ELEVATION_BAND_FEET = 500


# ============================================================
# HELPERS
# ============================================================


def get_project_paths():
    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    catalog_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_catalog.csv"
    )

    topography_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_topography_features.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "master_weather_groups.csv"
    )

    group_summary_path = (
        project_root
        / "data"
        / "processed"
        / "weather_group_summary.csv"
    )

    return (
        catalog_path,
        topography_path,
        output_path,
        group_summary_path,
    )


def assign_grid_bin(
    value,
    grid_size,
):
    """
    Convert a coordinate into a stable grid index.
    """

    return int(
        np.floor(
            value
            / grid_size
        )
    )


def assign_elevation_band(
    elevation_feet,
):
    """
    Convert mean trail elevation into a 500 ft band.
    """

    return int(
        np.floor(
            elevation_feet
            / ELEVATION_BAND_FEET
        )
    )


def main():

    (
        catalog_path,
        topography_path,
        output_path,
        group_summary_path,
    ) = get_project_paths()

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    print(
        "Loading master trail catalog..."
    )

    catalog = pd.read_csv(
        catalog_path
    )

    print(
        f"Catalog trails: "
        f"{len(catalog):,}"
    )

    print(
        "Loading topography features..."
    )

    topography = pd.read_csv(
        topography_path
    )

    print(
        f"Topography trails: "
        f"{len(topography):,}"
    )

    # ---------------------------------------------------------
    # MERGE ELEVATION
    # ---------------------------------------------------------

    trails = (
        catalog[
            [
                "trail_name",
                "final_area",
                "latitude",
                "longitude",
            ]
        ]
        .merge(
            topography[
                [
                    "trail_name",
                    "mean_elevation_feet",
                ]
            ],
            on="trail_name",
            how="left",
            validate="one_to_one",
        )
    )

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    duplicate_names = (
        trails[
            "trail_name"
        ]
        .duplicated()
        .sum()
    )

    missing_values = (
        trails[
            [
                "latitude",
                "longitude",
                "mean_elevation_feet",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    print()
    print(
        f"Duplicate trail names: "
        f"{duplicate_names:,}"
    )

    print(
        f"Trails missing location/elevation: "
        f"{missing_values:,}"
    )

    if duplicate_names > 0:

        raise RuntimeError(
            "Duplicate trail names found."
        )

    if missing_values > 0:

        raise RuntimeError(
            "Some trails are missing "
            "coordinates or elevation."
        )

    # ---------------------------------------------------------
    # ASSIGN GRID CELLS
    # ---------------------------------------------------------

    print()
    print(
        "Assigning weather grid cells..."
    )

    trails[
        "latitude_grid"
    ] = trails[
        "latitude"
    ].apply(
        lambda value: assign_grid_bin(
            value,
            LAT_GRID_SIZE,
        )
    )

    trails[
        "longitude_grid"
    ] = trails[
        "longitude"
    ].apply(
        lambda value: assign_grid_bin(
            value,
            LON_GRID_SIZE,
        )
    )

    trails[
        "elevation_band"
    ] = trails[
        "mean_elevation_feet"
    ].apply(
        assign_elevation_band
    )

    # ---------------------------------------------------------
    # CREATE GROUP KEY
    # ---------------------------------------------------------

    trails[
        "weather_group_key"
    ] = (
        trails[
            "latitude_grid"
        ].astype(str)
        + "_"
        + trails[
            "longitude_grid"
        ].astype(str)
        + "_"
        + trails[
            "elevation_band"
        ].astype(str)
    )

    # ---------------------------------------------------------
    # ASSIGN COMPACT GROUP IDS
    # ---------------------------------------------------------

    unique_keys = sorted(
        trails[
            "weather_group_key"
        ]
        .unique()
    )

    group_id_map = {
        key:
            f"WG{number:03d}"
        for number, key
        in enumerate(
            unique_keys,
            start=1,
        )
    }

    trails[
        "weather_group_id"
    ] = trails[
        "weather_group_key"
    ].map(
        group_id_map
    )

    # ---------------------------------------------------------
    # BUILD GROUP REPRESENTATIVE LOCATIONS
    # ---------------------------------------------------------
    #
    # Use mean latitude, longitude, and elevation for the
    # representative weather request.
    #

    summary = (
        trails
        .groupby(
            "weather_group_id",
            as_index=False,
        )
        .agg(
            representative_latitude=(
                "latitude",
                "mean",
            ),
            representative_longitude=(
                "longitude",
                "mean",
            ),
            representative_elevation_feet=(
                "mean_elevation_feet",
                "mean",
            ),
            trail_count=(
                "trail_name",
                "size",
            ),
            minimum_elevation_feet=(
                "mean_elevation_feet",
                "min",
            ),
            maximum_elevation_feet=(
                "mean_elevation_feet",
                "max",
            ),
        )
    )

    # ---------------------------------------------------------
    # ATTACH GROUP REPRESENTATIVE DATA TO EVERY TRAIL
    # ---------------------------------------------------------

    trails = trails.merge(
        summary,
        on="weather_group_id",
        how="left",
        validate="many_to_one",
    )

    # ---------------------------------------------------------
    # GROUP DIAGNOSTICS
    # ---------------------------------------------------------

    trails[
        "elevation_difference_from_group_feet"
    ] = (
        trails[
            "mean_elevation_feet"
        ]
        - trails[
            "representative_elevation_feet"
        ]
    ).abs()

    max_elevation_difference = (
        trails[
            "elevation_difference_from_group_feet"
        ]
        .max()
    )

    largest_group = (
        summary[
            "trail_count"
        ]
        .max()
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trails = trails.sort_values(
        [
            "weather_group_id",
            "trail_name",
        ]
    ).reset_index(
        drop=True
    )

    summary = summary.sort_values(
        [
            "trail_count",
            "weather_group_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    trails.to_csv(
        output_path,
        index=False,
    )

    summary.to_csv(
        group_summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MASTER WEATHER GROUP BUILD COMPLETE"
    )
    print("=" * 72)

    print(
        f"Trails grouped: "
        f"{len(trails):,}"
    )

    print(
        f"Weather groups: "
        f"{summary['weather_group_id'].nunique():,}"
    )

    print(
        f"Average trails per group: "
        f"{len(trails) / len(summary):.1f}"
    )

    print(
        f"Largest group size: "
        f"{largest_group:,}"
    )

    print(
        "Maximum trail-to-group elevation difference: "
        f"{max_elevation_difference:.0f} ft"
    )

    print()
    print(
        f"Saved trail-to-weather mapping to:"
        f"\n  {output_path}"
    )

    print()
    print(
        f"Saved weather group summary to:"
        f"\n  {group_summary_path}"
    )

    # ---------------------------------------------------------
    # GROUP SIZE DISTRIBUTION
    # ---------------------------------------------------------

    print()
    print(
        "Group size distribution:"
    )

    print()

    print(
        summary[
            "trail_count"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "trails_in_group"
        )
        .to_string()
    )

    # ---------------------------------------------------------
    # LARGEST GROUPS
    # ---------------------------------------------------------

    print()
    print(
        "Largest 20 weather groups:"
    )

    print()

    print(
        summary[
            [
                "weather_group_id",
                "trail_count",
                "representative_latitude",
                "representative_longitude",
                "representative_elevation_feet",
                "minimum_elevation_feet",
                "maximum_elevation_feet",
            ]
        ]
        .head(20)
        .round(
            {
                "representative_latitude": 5,
                "representative_longitude": 5,
                "representative_elevation_feet": 0,
                "minimum_elevation_feet": 0,
                "maximum_elevation_feet": 0,
            }
        )
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # EXAMPLE TRAILS FROM LARGEST GROUP
    # ---------------------------------------------------------

    largest_group_id = (
        summary.iloc[0][
            "weather_group_id"
        ]
    )

    largest_group_trails = (
        trails[
            trails[
                "weather_group_id"
            ]
            == largest_group_id
        ]
        .sort_values(
            "mean_elevation_feet"
        )
    )

    print()
    print(
        f"Example largest group "
        f"({largest_group_id}):"
    )

    print()

    print(
        largest_group_trails[
            [
                "trail_name",
                "final_area",
                "latitude",
                "longitude",
                "mean_elevation_feet",
                "elevation_difference_from_group_feet",
            ]
        ]
        .round(
            {
                "latitude": 5,
                "longitude": 5,
                "mean_elevation_feet": 0,
                "elevation_difference_from_group_feet": 0,
            }
        )
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()