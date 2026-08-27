from pathlib import Path

import osmnx as ox
import pandas as pd


# ============================================================
# STUDY AREA
# ============================================================
#
# OSMnx 2.x bbox order:
#
# (left, bottom, right, top)
#
# This first-pass box is intentionally generous so we can inspect
# the full Park City-area riding network before deciding what to keep.
#

BBOX = (
    -111.680,   # left / west
    40.585,     # bottom / south
    -111.400,   # right / east
    40.790,     # top / north
)


# Trail-like highway classifications we want to inspect.
ALLOWED_HIGHWAY_TYPES = {
    "path",
    "track",
}


def normalize_highway(value):
    """
    OSMnx attributes may occasionally contain lists.
    Convert highway values into a simple comparable form.
    """

    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    return [value]


def normalize_name(value):
    """
    Clean OSM trail names while preserving readable names.
    """

    if isinstance(value, list):
        value = " / ".join(
            str(item)
            for item in value
            if item
        )

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def main():
    project_root = Path(__file__).resolve().parents[2]

    raw_graph_path = (
        project_root
        / "data"
        / "raw"
        / "expanded_park_city_bike_network.graphml"
    )

    raw_edges_path = (
        project_root
        / "data"
        / "raw"
        / "expanded_park_city_edges.geojson"
    )

    candidate_path = (
        project_root
        / "data"
        / "processed"
        / "expanded_trail_candidates.csv"
    )

    summary_path = (
        project_root
        / "data"
        / "processed"
        / "expanded_trail_name_summary.csv"
    )

    # ---------------------------------------------------------
    # DOWNLOAD NETWORK
    # ---------------------------------------------------------

    print("Downloading expanded Park City bicycle network...")

    print(
        "Bounding box:"
        f"\n  west:  {BBOX[0]}"
        f"\n  south: {BBOX[1]}"
        f"\n  east:  {BBOX[2]}"
        f"\n  north: {BBOX[3]}"
    )

    graph = ox.graph.graph_from_bbox(
        bbox=BBOX,
        network_type="bike",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    print()
    print("Download complete.")

    print(
        f"Graph nodes: {len(graph.nodes):,}"
    )

    print(
        f"Graph edges: {len(graph.edges):,}"
    )

    # ---------------------------------------------------------
    # SAVE RAW GRAPH
    # ---------------------------------------------------------

    raw_graph_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ox.io.save_graphml(
        graph,
        filepath=raw_graph_path,
    )

    # ---------------------------------------------------------
    # CONVERT TO GEODATAFRAME
    # ---------------------------------------------------------

    nodes, edges = ox.graph_to_gdfs(
        graph
    )

    edges = edges.reset_index()

    print()
    print(
        f"Edge rows: {len(edges):,}"
    )

    # Save raw edges for later mapping/inspection.
    edges.to_file(
        raw_edges_path,
        driver="GeoJSON",
    )

    # ---------------------------------------------------------
    # NORMALIZE HIGHWAY + NAME
    # ---------------------------------------------------------

    edges["trail_name"] = (
        edges["name"]
        .apply(normalize_name)
    )

    edges["highway_values"] = (
        edges["highway"]
        .apply(normalize_highway)
    )

    # ---------------------------------------------------------
    # FILTER TO TRAIL-LIKE SEGMENTS
    # ---------------------------------------------------------

    def is_allowed_highway(values):
        return any(
            value in ALLOWED_HIGHWAY_TYPES
            for value in values
        )

    trail_edges = edges[
        edges["highway_values"]
        .apply(is_allowed_highway)
    ].copy()

    print(
        f"Path/track segments: "
        f"{len(trail_edges):,}"
    )

    # ---------------------------------------------------------
    # KEEP NAMED TRAILS ONLY
    # ---------------------------------------------------------

    named = trail_edges[
        trail_edges["trail_name"]
        .notna()
    ].copy()

    print(
        f"Named path/track segments: "
        f"{len(named):,}"
    )

    # ---------------------------------------------------------
    # CALCULATE MILES
    # ---------------------------------------------------------

    named["length_miles"] = (
        named["length"]
        / 1609.344
    )

    # ---------------------------------------------------------
    # SAVE SEGMENT-LEVEL CANDIDATES
    # ---------------------------------------------------------

    candidate_columns = [
        column
        for column in [
            "u",
            "v",
            "key",
            "osmid",
            "trail_name",
            "highway",
            "length",
            "length_miles",
            "access",
            "bicycle",
            "foot",
            "surface",
            "tracktype",
            "smoothness",
            "mtb:scale",
            "mtb:scale:uphill",
            "sac_scale",
        ]
        if column in named.columns
    ]

    candidates = named[
        candidate_columns
    ].copy()

    candidate_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        candidate_path,
        index=False,
    )

    # ---------------------------------------------------------
    # BUILD NAME SUMMARY
    # ---------------------------------------------------------

    summary = (
        named
        .groupby(
            "trail_name",
            as_index=False,
        )
        .agg(
            distance_miles=(
                "length_miles",
                "sum",
            ),
            segment_count=(
                "trail_name",
                "size",
            ),
        )
    )

    summary = summary.sort_values(
        [
            "distance_miles",
            "trail_name",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EXPANDED TRAIL DISCOVERY COMPLETE"
    )
    print("=" * 70)

    print(
        f"Named candidate trails: "
        f"{summary['trail_name'].nunique():,}"
    )

    print(
        f"Candidate segments: "
        f"{len(candidates):,}"
    )

    print()
    print(
        f"Saved raw graph to:"
        f"\n  {raw_graph_path}"
    )

    print()
    print(
        f"Saved raw edge geometry to:"
        f"\n  {raw_edges_path}"
    )

    print()
    print(
        f"Saved candidate segments to:"
        f"\n  {candidate_path}"
    )

    print()
    print(
        f"Saved trail-name summary to:"
        f"\n  {summary_path}"
    )

    print()
    print("Longest 50 named candidates:")
    print()

    print(
        summary.head(50)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()