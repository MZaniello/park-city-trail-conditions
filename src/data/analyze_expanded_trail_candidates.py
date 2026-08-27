from pathlib import Path

import networkx as nx
import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[2]

    input_path = (
        project_root
        / "data"
        / "processed"
        / "expanded_trail_candidates.csv"
    )

    segment_output_path = (
        project_root
        / "data"
        / "processed"
        / "expanded_trail_segments_deduplicated.csv"
    )

    component_output_path = (
        project_root
        / "data"
        / "processed"
        / "expanded_trail_components.csv"
    )

    # ---------------------------------------------------------
    # LOAD
    # ---------------------------------------------------------

    print("Loading expanded trail candidates...")

    trails = pd.read_csv(
        input_path
    )

    print(
        f"Raw candidate segments: "
        f"{len(trails):,}"
    )

    print(
        f"Raw trail names: "
        f"{trails['trail_name'].nunique():,}"
    )

    # ---------------------------------------------------------
    # NORMALIZE ENDPOINTS
    # ---------------------------------------------------------
    #
    # A physical edge may appear as:
    #
    #   u -> v
    #   v -> u
    #
    # Treat those as the same segment.
    #

    trails["node_a"] = trails[
        ["u", "v"]
    ].min(axis=1)

    trails["node_b"] = trails[
        ["u", "v"]
    ].max(axis=1)

    # Rounded length gives us a little protection against
    # tiny floating-point differences.
    trails["length_rounded"] = (
        trails["length"]
        .round(2)
    )

    # ---------------------------------------------------------
    # REMOVE REVERSE-DIRECTION DUPLICATES
    # ---------------------------------------------------------

    rows_before = len(trails)

    deduplicated = (
        trails
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
        - len(deduplicated)
    )

    deduplicated["length_miles"] = (
        deduplicated["length"]
        / 1609.344
    )

    print()
    print(
        f"Reverse/directional duplicates removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Physical candidate segments remaining: "
        f"{len(deduplicated):,}"
    )

    # ---------------------------------------------------------
    # SAVE DEDUPLICATED SEGMENTS
    # ---------------------------------------------------------

    deduplicated.to_csv(
        segment_output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SPLIT EACH NAME INTO CONNECTED COMPONENTS
    # ---------------------------------------------------------
    #
    # If the same trail name occurs in disconnected pieces,
    # don't automatically treat all of them as one trail.
    #

    print()
    print(
        "Finding connected components "
        "within each trail name..."
    )

    component_rows = []

    for trail_name, group in deduplicated.groupby(
        "trail_name"
    ):

        graph = nx.Graph()

        for _, row in group.iterrows():

            graph.add_edge(
                int(row["u"]),
                int(row["v"]),
            )

        components = list(
            nx.connected_components(graph)
        )

        # Largest components first.
        component_data = []

        for component_number, nodes in enumerate(
            components,
            start=1,
        ):

            component_segments = group[
                group["u"].isin(nodes)
                & group["v"].isin(nodes)
            ].copy()

            distance_miles = (
                component_segments[
                    "length_miles"
                ].sum()
            )

            component_data.append(
                {
                    "trail_name": trail_name,
                    "component_id": component_number,
                    "distance_miles": distance_miles,
                    "segment_count": len(
                        component_segments
                    ),
                    "node_count": len(nodes),
                }
            )

        component_data = sorted(
            component_data,
            key=lambda x: x[
                "distance_miles"
            ],
            reverse=True,
        )

        # Re-number after sorting.
        for number, record in enumerate(
            component_data,
            start=1,
        ):

            record[
                "component_id"
            ] = number

            record[
                "is_largest_component"
            ] = (
                number == 1
            )

            component_rows.append(
                record
            )

    components = pd.DataFrame(
        component_rows
    )

    components = components.sort_values(
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

    components.to_csv(
        component_output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # DIAGNOSTICS
    # ---------------------------------------------------------

    component_counts = (
        components
        .groupby("trail_name")
        .size()
    )

    names_with_multiple_components = (
        component_counts > 1
    ).sum()

    largest_only = components[
        components[
            "is_largest_component"
        ]
    ].copy()

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "EXPANDED TRAIL COMPONENT ANALYSIS COMPLETE"
    )
    print("=" * 72)

    print(
        f"Unique trail names: "
        f"{components['trail_name'].nunique():,}"
    )

    print(
        f"Total connected components: "
        f"{len(components):,}"
    )

    print(
        f"Names with multiple disconnected components: "
        f"{names_with_multiple_components:,}"
    )

    print()
    print(
        f"Saved deduplicated segments to:"
        f"\n  {segment_output_path}"
    )

    print()
    print(
        f"Saved connected-component summary to:"
        f"\n  {component_output_path}"
    )

    print()
    print(
        "Longest 60 largest connected components:"
    )

    print()

    print(
        largest_only[
            [
                "trail_name",
                "distance_miles",
                "segment_count",
                "node_count",
            ]
        ]
        .head(60)
        .to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # SHOW HEAVILY FRAGMENTED NAMES
    # ---------------------------------------------------------

    fragmented = (
        component_counts[
            component_counts > 1
        ]
        .sort_values(
            ascending=False
        )
        .head(30)
    )

    if not fragmented.empty:

        print()
        print(
            "Most fragmented trail names:"
        )

        print()

        print(
            fragmented.to_string()
        )


if __name__ == "__main__":
    main()