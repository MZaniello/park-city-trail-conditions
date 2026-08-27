from pathlib import Path
import ast

import geopandas as gpd
import networkx as nx
import pandas as pd


def normalize_name(value):
    """
    Normalize OSM trail names from strings, lists,
    NumPy arrays, or serialized lists.
    """

    if isinstance(value, (list, tuple)) or (
        hasattr(value, "tolist")
        and not isinstance(value, str)
    ):

        if hasattr(value, "tolist"):
            value = value.tolist()

        if not isinstance(value, (list, tuple)):
            value = [value]

        values = []

        for item in value:
            if item is None:
                continue

            item = str(item).strip()

            if item:
                values.append(item)

        if not values:
            return None

        return " / ".join(values)

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (ValueError, TypeError):
        pass

    value = str(value).strip()

    if (
        value.startswith("[")
        and value.endswith("]")
    ):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple)):
                parsed = [
                    str(item).strip()
                    for item in parsed
                    if item is not None
                    and str(item).strip()
                ]

                if parsed:
                    return " / ".join(parsed)

        except (ValueError, SyntaxError):
            pass

    if not value:
        return None

    return value


def normalize_highway(value):
    """
    Normalize OSM highway values.
    """

    if isinstance(value, (list, tuple)) or (
        hasattr(value, "tolist")
        and not isinstance(value, str)
    ):

        if hasattr(value, "tolist"):
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            return [
                str(item).strip()
                for item in value
                if item is not None
                and str(item).strip()
            ]

        return [str(value).strip()]

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (ValueError, TypeError):
        pass

    value = str(value).strip()

    if (
        value.startswith("[")
        and value.endswith("]")
    ):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple)):
                return [
                    str(item).strip()
                    for item in parsed
                    if item is not None
                    and str(item).strip()
                ]

        except (ValueError, SyntaxError):
            pass

    if not value:
        return []

    return [value]


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

    catalog_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_catalog.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_geometries.geojson"
    )

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    print("Loading expanded OSM geometry...")

    edges = gpd.read_file(edges_path)

    print(
        f"Raw edges: {len(edges):,}"
    )

    print("Loading master trail catalog...")

    catalog = pd.read_csv(
        catalog_path,
        keep_default_na=False,
    )

    included_names = set(
        catalog["trail_name"]
    )

    print(
        f"Included catalog trails: "
        f"{len(included_names):,}"
    )

    # ---------------------------------------------------------
    # NORMALIZE OSM FIELDS
    # ---------------------------------------------------------

    edges["trail_name"] = (
        edges["name"]
        .apply(normalize_name)
    )

    edges["highway_values"] = (
        edges["highway"]
        .apply(normalize_highway)
    )

    allowed_types = {
        "path",
        "track",
    }

    edges = edges[
        edges["highway_values"]
        .apply(
            lambda values: any(
                highway in allowed_types
                for highway in values
            )
        )
    ].copy()

    edges = edges[
        edges["trail_name"].isin(
            included_names
        )
    ].copy()

    print(
        f"Included path/track edge rows: "
        f"{len(edges):,}"
    )

    # ---------------------------------------------------------
    # NUMERIC CLEANUP
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

    edges = edges.dropna(
        subset=[
            "u",
            "v",
            "length",
        ]
    ).copy()

    # ---------------------------------------------------------
    # REMOVE DIRECTIONAL DUPLICATES
    # ---------------------------------------------------------

    edges["node_a"] = (
        edges[["u", "v"]]
        .min(axis=1)
    )

    edges["node_b"] = (
        edges[["u", "v"]]
        .max(axis=1)
    )

    edges["length_rounded"] = (
        edges["length"]
        .round(2)
    )

    rows_before = len(edges)

    edges = edges.drop_duplicates(
        subset=[
            "trail_name",
            "node_a",
            "node_b",
            "length_rounded",
        ]
    ).copy()

    print(
        "Directional duplicates removed: "
        f"{rows_before - len(edges):,}"
    )

    print(
        f"Physical included segments: "
        f"{len(edges):,}"
    )

    # ---------------------------------------------------------
    # LARGEST CONNECTED COMPONENT PER TRAIL
    # ---------------------------------------------------------

    print()
    print(
        "Building master trail geometries..."
    )

    records = []

    grouped = edges.groupby(
        "trail_name"
    )

    total = edges[
        "trail_name"
    ].nunique()

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
            or number == total
        ):
            print(
                f"  Processing "
                f"{number}/{total}..."
            )

        graph = nx.Graph()

        for _, row in group.iterrows():

            graph.add_edge(
                int(row["u"]),
                int(row["v"]),
            )

        components = list(
            nx.connected_components(
                graph
            )
        )

        best_group = None
        best_length = -1

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

            if total_length > best_length:

                best_length = (
                    total_length
                )

                best_group = (
                    component_edges
                )

        if best_group is None:
            continue

        geometry = (
            best_group.geometry
            .union_all()
        )

        records.append(
            {
                "trail_name":
                    trail_name,

                "geometry":
                    geometry,

                "geometry_length_miles":
                    best_length
                    / 1609.344,

                "geometry_segment_count":
                    len(best_group),
            }
        )

    geometries = gpd.GeoDataFrame(
        records,
        geometry="geometry",
        crs=edges.crs,
    )

    # ---------------------------------------------------------
    # JOIN CATALOG METADATA
    # ---------------------------------------------------------

    geometries = geometries.merge(
        catalog,
        on="trail_name",
        how="left",
        validate="one_to_one",
    )

    geometries = geometries.sort_values(
        [
            "final_area",
            "trail_name",
        ]
    ).reset_index(
        drop=True
    )

    # ---------------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------------

    missing_geometry_names = (
        included_names
        - set(
            geometries[
                "trail_name"
            ]
        )
    )

    duplicates = (
        geometries[
            "trail_name"
        ]
        .duplicated()
        .sum()
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    geometries.to_file(
        output_path,
        driver="GeoJSON",
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MASTER TRAIL GEOMETRY BUILD COMPLETE"
    )
    print("=" * 72)

    print(
        f"Catalog trails: "
        f"{len(included_names):,}"
    )

    print(
        f"Geometries created: "
        f"{len(geometries):,}"
    )

    print(
        f"Duplicate trail names: "
        f"{duplicates:,}"
    )

    print(
        f"Missing geometries: "
        f"{len(missing_geometry_names):,}"
    )

    if missing_geometry_names:

        print()
        print(
            "Missing trail names:"
        )

        for trail_name in sorted(
            missing_geometry_names
        ):
            print(
                f"  - {trail_name}"
            )

    print()
    print(
        f"Saved to:"
        f"\n  {output_path}"
    )

    print()
    print(
        "Area counts:"
    )

    print()

    print(
        geometries[
            "final_area"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Longest 25 included trails:"
    )

    print()

    print(
        geometries[
            [
                "trail_name",
                "final_area",
                "geometry_length_miles",
            ]
        ]
        .sort_values(
            "geometry_length_miles",
            ascending=False,
        )
        .head(25)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
