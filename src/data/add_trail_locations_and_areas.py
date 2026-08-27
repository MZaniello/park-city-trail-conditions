from pathlib import Path
import ast

import geopandas as gpd
import networkx as nx
import pandas as pd


# ============================================================
# APPROXIMATE AREA CENTERS
# ============================================================
#
# These are only used to create a first-pass area_guess for
# catalog review. They are NOT treated as authoritative
# geographic boundaries.
#
# Format:
#     area_name: (latitude, longitude)
#

AREA_CENTERS = {
    "Summit Park": (
        40.743,
        -111.611,
    ),
    "Pinebrook": (
        40.738,
        -111.566,
    ),
    "Jeremy Ranch / Glenwild": (
        40.724,
        -111.553,
    ),
    "Park City / PCMR": (
        40.651,
        -111.515,
    ),
    "Deer Valley": (
        40.625,
        -111.490,
    ),
    "Round Valley": (
        40.688,
        -111.475,
    ),
    "Clark Ranch": (
        40.700,
        -111.430,
    ),
    "Jordanelle": (
        40.635,
        -111.420,
    ),
    "Wasatch Crest / Millcreek": (
        40.690,
        -111.640,
    ),
}


def normalize_name(value):
    """
    Normalize OSM trail names.

    Handles:
    - strings
    - Python lists
    - tuples
    - NumPy arrays
    - serialized list strings
    - missing values
    """

    # ---------------------------------------------------------
    # ACTUAL LIST-LIKE VALUES
    # ---------------------------------------------------------

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ) or (
        hasattr(value, "tolist")
        and not isinstance(value, str)
    ):

        if hasattr(
            value,
            "tolist",
        ):
            value = value.tolist()

        if not isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            value = [value]

        values = []

        for item in value:

            if item is None:
                continue

            item = str(
                item
            ).strip()

            if item:
                values.append(
                    item
                )

        if not values:
            return None

        return " / ".join(
            values
        )

    # ---------------------------------------------------------
    # SCALAR MISSING VALUES
    # ---------------------------------------------------------

    if value is None:
        return None

    try:

        if pd.isna(
            value
        ):
            return None

    except (
        ValueError,
        TypeError,
    ):
        pass

    value = str(
        value
    ).strip()

    # ---------------------------------------------------------
    # SERIALIZED LIST STRINGS
    # ---------------------------------------------------------
    #
    # Example:
    #
    # "['Trail A', 'Trail B']"
    #

    if (
        value.startswith("[")
        and value.endswith("]")
    ):

        try:

            parsed = ast.literal_eval(
                value
            )

            if isinstance(
                parsed,
                (
                    list,
                    tuple,
                ),
            ):

                parsed = [
                    str(item).strip()
                    for item in parsed
                    if item is not None
                    and str(item).strip()
                ]

                if parsed:

                    return " / ".join(
                        parsed
                    )

        except (
            ValueError,
            SyntaxError,
        ):
            pass

    if not value:
        return None

    return value


def normalize_highway(value):
    """
    Convert OSM highway attributes into a Python list.

    Handles:
    - strings
    - lists
    - tuples
    - NumPy arrays
    - serialized list strings
    - missing values
    """

    # ---------------------------------------------------------
    # ACTUAL LIST-LIKE VALUES
    # ---------------------------------------------------------

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ) or (
        hasattr(value, "tolist")
        and not isinstance(value, str)
    ):

        if hasattr(
            value,
            "tolist",
        ):
            value = value.tolist()

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):

            return [
                str(item).strip()
                for item in value
                if item is not None
                and str(item).strip()
            ]

        value = str(
            value
        ).strip()

        if not value:
            return []

        return [
            value
        ]

    # ---------------------------------------------------------
    # SCALAR MISSING VALUES
    # ---------------------------------------------------------

    if value is None:
        return []

    try:

        if pd.isna(
            value
        ):
            return []

    except (
        ValueError,
        TypeError,
    ):
        pass

    value = str(
        value
    ).strip()

    # ---------------------------------------------------------
    # SERIALIZED LIST STRINGS
    # ---------------------------------------------------------

    if (
        value.startswith("[")
        and value.endswith("]")
    ):

        try:

            parsed = ast.literal_eval(
                value
            )

            if isinstance(
                parsed,
                (
                    list,
                    tuple,
                ),
            ):

                return [
                    str(item).strip()
                    for item in parsed
                    if item is not None
                    and str(item).strip()
                ]

        except (
            ValueError,
            SyntaxError,
        ):
            pass

    if not value:
        return []

    return [
        value
    ]


def nearest_area(
    latitude,
    longitude,
):
    """
    Assign the closest approximate riding-area center.

    Uses simple squared latitude/longitude distance because this
    is only a first-pass review label across a relatively small
    geographic region.
    """

    best_area = None
    best_distance = None

    for area, (
        area_latitude,
        area_longitude,
    ) in AREA_CENTERS.items():

        distance = (
            (
                latitude
                - area_latitude
            ) ** 2
            + (
                longitude
                - area_longitude
            ) ** 2
        )

        if (
            best_distance is None
            or distance
            < best_distance
        ):

            best_area = area
            best_distance = (
                distance
            )

    return best_area


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    edges_path = (
        project_root
        / "data"
        / "raw"
        / "expanded_park_city_edges.geojson"
    )

    review_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_review.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_review_with_locations.csv"
    )

    locations_path = (
        project_root
        / "data"
        / "processed"
        / "expanded_trail_locations.csv"
    )

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    print(
        "Loading expanded trail geometry..."
    )

    edges = gpd.read_file(
        edges_path
    )

    print(
        f"Raw geometry edges: "
        f"{len(edges):,}"
    )

    print(
        "Loading master review table..."
    )

    review = pd.read_csv(
        review_path,
        keep_default_na=False,
    )

    print(
        f"Review trail names: "
        f"{len(review):,}"
    )

    # ---------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # ---------------------------------------------------------

    required_edge_columns = {
        "u",
        "v",
        "name",
        "highway",
        "length",
        "geometry",
    }

    missing_edge_columns = (
        required_edge_columns
        - set(
            edges.columns
        )
    )

    if missing_edge_columns:

        raise ValueError(
            "Expanded edge file is missing required columns: "
            + ", ".join(
                sorted(
                    missing_edge_columns
                )
            )
        )

    if "trail_name" not in review.columns:

        raise ValueError(
            "master_trail_review.csv is missing "
            "the trail_name column."
        )

    # ---------------------------------------------------------
    # NORMALIZE OSM ATTRIBUTES
    # ---------------------------------------------------------

    edges["trail_name"] = (
        edges["name"]
        .apply(
            normalize_name
        )
    )

    edges["highway_values"] = (
        edges["highway"]
        .apply(
            normalize_highway
        )
    )

    allowed_types = {
        "path",
        "track",
    }

    edges = edges[
        edges[
            "highway_values"
        ]
        .apply(
            lambda values: any(
                highway
                in allowed_types
                for highway
                in values
            )
        )
    ].copy()

    edges = edges[
        edges[
            "trail_name"
        ]
        .notna()
    ].copy()

    print(
        f"Named path/track edges: "
        f"{len(edges):,}"
    )

    # ---------------------------------------------------------
    # ENSURE NODE IDS ARE NUMERIC
    # ---------------------------------------------------------

    edges["u"] = pd.to_numeric(
        edges["u"],
        errors="coerce",
    )

    edges["v"] = pd.to_numeric(
        edges["v"],
        errors="coerce",
    )

    edges["length"] = pd.to_numeric(
        edges["length"],
        errors="coerce",
    )

    invalid_edge_rows = (
        edges[
            [
                "u",
                "v",
                "length",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    )

    invalid_count = (
        invalid_edge_rows.sum()
    )

    if invalid_count:

        print(
            f"Removing "
            f"{invalid_count:,} edge rows "
            "with invalid u/v/length values..."
        )

        edges = edges[
            ~invalid_edge_rows
        ].copy()

    # ---------------------------------------------------------
    # DEDUPLICATE PHYSICAL SEGMENTS
    # ---------------------------------------------------------

    edges["node_a"] = (
        edges[
            [
                "u",
                "v",
            ]
        ]
        .min(
            axis=1
        )
    )

    edges["node_b"] = (
        edges[
            [
                "u",
                "v",
            ]
        ]
        .max(
            axis=1
        )
    )

    edges[
        "length_rounded"
    ] = (
        edges[
            "length"
        ]
        .round(2)
    )

    rows_before = len(
        edges
    )

    edges = (
        edges
        .drop_duplicates(
            subset=[
                "trail_name",
                "node_a",
                "node_b",
                "length_rounded",
            ]
        )
        .copy()
    )

    duplicates_removed = (
        rows_before
        - len(
            edges
        )
    )

    print(
        f"Directional duplicates removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Physical named segments: "
        f"{len(edges):,}"
    )

    # ---------------------------------------------------------
    # FIND LARGEST CONNECTED COMPONENT FOR EACH NAME
    # ---------------------------------------------------------

    print()
    print(
        "Calculating largest trail components..."
    )

    trail_geometries = []

    grouped = edges.groupby(
        "trail_name"
    )

    total_names = (
        edges[
            "trail_name"
        ]
        .nunique()
    )

    for number, (
        trail_name,
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):

        if (
            number == 1
            or number % 50 == 0
            or number == total_names
        ):

            print(
                f"  Processing trail "
                f"{number:,}/{total_names:,}..."
            )

        graph = nx.Graph()

        for _, row in group.iterrows():

            graph.add_edge(
                int(
                    row["u"]
                ),
                int(
                    row["v"]
                ),
            )

        components = list(
            nx.connected_components(
                graph
            )
        )

        if not components:
            continue

        best_component = None
        best_group = None
        best_length = -1.0

        for component in components:

            component_edges = group[
                group["u"].isin(
                    component
                )
                & group["v"].isin(
                    component
                )
            ].copy()

            total_length = (
                component_edges[
                    "length"
                ].sum()
            )

            if (
                total_length
                > best_length
            ):

                best_length = (
                    total_length
                )

                best_component = (
                    component
                )

                best_group = (
                    component_edges
                )

        if (
            best_group is None
            or best_group.empty
        ):
            continue

        geometry = (
            best_group[
                "geometry"
            ]
            .union_all()
        )

        trail_geometries.append(
            {
                "trail_name":
                    trail_name,

                "geometry":
                    geometry,

                "distance_miles":
                    best_length
                    / 1609.344,

                "component_node_count":
                    len(
                        best_component
                    ),

                "component_segment_count":
                    len(
                        best_group
                    ),
            }
        )

    trails = gpd.GeoDataFrame(
        trail_geometries,
        geometry="geometry",
        crs=edges.crs,
    )

    print()
    print(
        f"Largest trail geometries: "
        f"{len(trails):,}"
    )

    if trails.empty:

        raise ValueError(
            "No trail geometries were created."
        )

    # ---------------------------------------------------------
    # CALCULATE CENTROIDS
    # ---------------------------------------------------------
    #
    # Use projected coordinates for centroid calculation.
    #
    # EPSG:26912 = NAD83 / UTM zone 12N
    # Appropriate for northern Utah.
    #

    print(
        "Calculating trail centroids..."
    )

    projected = (
        trails
        .to_crs(
            "EPSG:26912"
        )
        .copy()
    )

    centroid_geometry = (
        projected.geometry
        .centroid
    )

    centroid_gdf = (
        gpd.GeoDataFrame(
            projected[
                [
                    "trail_name"
                ]
            ].copy(),
            geometry=centroid_geometry,
            crs=projected.crs,
        )
        .to_crs(
            "EPSG:4326"
        )
    )

    trails["latitude"] = (
        centroid_gdf.geometry.y.values
    )

    trails["longitude"] = (
        centroid_gdf.geometry.x.values
    )

    # ---------------------------------------------------------
    # ASSIGN AREA GUESS
    # ---------------------------------------------------------

    print(
        "Assigning approximate riding areas..."
    )

    trails[
        "area_guess"
    ] = trails.apply(
        lambda row: nearest_area(
            row[
                "latitude"
            ],
            row[
                "longitude"
            ],
        ),
        axis=1,
    )

    # ---------------------------------------------------------
    # SAVE TRAIL LOCATION TABLE
    # ---------------------------------------------------------

    locations = trails[
        [
            "trail_name",
            "latitude",
            "longitude",
            "distance_miles",
            "component_segment_count",
            "component_node_count",
            "area_guess",
        ]
    ].copy()

    locations = (
        locations
        .sort_values(
            [
                "area_guess",
                "trail_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    locations_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    locations.to_csv(
        locations_path,
        index=False,
    )

    # ---------------------------------------------------------
    # MERGE INTO REVIEW TABLE
    # ---------------------------------------------------------

    enriched = review.merge(
        locations[
            [
                "trail_name",
                "latitude",
                "longitude",
                "area_guess",
            ]
        ],
        on="trail_name",
        how="left",
        validate="one_to_one",
    )

    # Preserve:
    #
    # area_guess = automatic first pass
    # area       = final human-reviewed area
    #

    enriched.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    missing_locations = (
        enriched[
            "latitude"
        ]
        .isna()
        .sum()
    )

    area_counts = (
        enriched[
            "area_guess"
        ]
        .replace(
            "",
            pd.NA,
        )
        .dropna()
        .value_counts()
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "TRAIL LOCATION / AREA ANALYSIS COMPLETE"
    )
    print("=" * 72)

    print(
        f"Trails in review table: "
        f"{len(enriched):,}"
    )

    print(
        f"Trails with locations: "
        f"{len(enriched) - missing_locations:,}"
    )

    print(
        f"Missing locations: "
        f"{missing_locations:,}"
    )

    print()
    print(
        "Approximate area counts:"
    )

    print()

    if area_counts.empty:

        print(
            "No area assignments found."
        )

    else:

        print(
            area_counts.to_string()
        )

    print()
    print(
        f"Saved locations to:"
        f"\n  {locations_path}"
    )

    print()
    print(
        f"Saved enriched review table to:"
        f"\n  {output_path}"
    )

    # ---------------------------------------------------------
    # EXAMPLE TRAILS BY AREA
    # ---------------------------------------------------------

    print()
    print(
        "Example trails by area:"
    )

    valid_areas = (
        enriched[
            "area_guess"
        ]
        .replace(
            "",
            pd.NA,
        )
        .dropna()
        .unique()
    )

    for area in sorted(
        valid_areas
    ):

        sample = (
            enriched[
                enriched[
                    "area_guess"
                ]
                == area
            ]
            .sort_values(
                "distance_miles",
                ascending=False,
            )[
                "trail_name"
            ]
            .head(10)
            .tolist()
        )

        print()
        print(
            f"{area}:"
        )

        for trail in sample:

            print(
                f"  - {trail}"
            )


if __name__ == "__main__":
    main()