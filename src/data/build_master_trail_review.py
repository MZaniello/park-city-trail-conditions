from pathlib import Path

import pandas as pd


# -------------------------------------------------------------
# OBVIOUS NAME-BASED EXCLUSIONS
# -------------------------------------------------------------
#
# These are conservative. We only auto-exclude names that are
# very likely not singletrack / MTB trail products we want to
# forecast individually.
#

EXCLUDE_KEYWORDS = [
    "service",
    "access",
    "drive",
    "road",
    "rail trail",
    "connector road",
    "parking",
    "ski run",
    "lift",
]


# Some names are useful trails even though they might contain
# words that sound road-like. Add exceptions here if needed.
KEEP_EXCEPTIONS = {
    "Road to WOS",
}


def should_auto_exclude(trail_name):
    """
    Conservative automatic name filter.
    """

    if trail_name in KEEP_EXCEPTIONS:
        return False

    lower = trail_name.lower()

    return any(
        keyword in lower
        for keyword in EXCLUDE_KEYWORDS
    )


def main():
    project_root = Path(__file__).resolve().parents[2]

    components_path = (
        project_root
        / "data"
        / "processed"
        / "expanded_trail_components.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_review.csv"
    )

    print("Loading trail components...")

    components = pd.read_csv(
        components_path
    )

    # ---------------------------------------------------------
    # KEEP LARGEST COMPONENT PER NAME FOR FIRST REVIEW PASS
    # ---------------------------------------------------------
    #
    # We are NOT deleting secondary components permanently.
    # This just keeps the review table manageable.
    #

    largest = components[
        components[
            "is_largest_component"
        ]
        == True
    ].copy()

    print(
        f"Unique trail names: "
        f"{largest['trail_name'].nunique():,}"
    )

    # ---------------------------------------------------------
    # BASIC QUALITY FLAGS
    # ---------------------------------------------------------

    largest["auto_exclude"] = (
        largest["trail_name"]
        .apply(
            should_auto_exclude
        )
    )

    # Extremely tiny named pieces are often mapping fragments,
    # connectors, or incomplete geometry.
    largest[
        "very_short"
    ] = (
        largest[
            "distance_miles"
        ]
        < 0.10
    )

    # ---------------------------------------------------------
    # INITIAL REVIEW DECISION
    # ---------------------------------------------------------
    #
    # Blank = human review still needed
    # INCLUDE = looks usable
    # EXCLUDE = obvious junk
    #

    largest["review_status"] = ""

    obvious_exclude = (
        largest["auto_exclude"]
        | largest["very_short"]
    )

    largest.loc[
        obvious_exclude,
        "review_status",
    ] = "EXCLUDE"

    # ---------------------------------------------------------
    # AREA LABEL PLACEHOLDER
    # ---------------------------------------------------------
    #
    # We'll populate this more intelligently next using geometry
    # / centroid location.
    #

    largest["area"] = ""

    # ---------------------------------------------------------
    # SOURCE + NOTES
    # ---------------------------------------------------------

    largest["source"] = "OpenStreetMap"

    largest["review_notes"] = ""

    # ---------------------------------------------------------
    # COLUMN ORDER
    # ---------------------------------------------------------

    review = largest[
        [
            "trail_name",
            "distance_miles",
            "segment_count",
            "node_count",
            "component_id",
            "auto_exclude",
            "very_short",
            "review_status",
            "area",
            "source",
            "review_notes",
        ]
    ].copy()

    # Put likely keepers first.
    review = review.sort_values(
        [
            "review_status",
            "distance_miles",
            "trail_name",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    excluded = (
        review[
            "review_status"
        ]
        == "EXCLUDE"
    ).sum()

    needs_review = (
        review[
            "review_status"
        ]
        == ""
    ).sum()

    print()
    print("=" * 72)
    print(
        "MASTER TRAIL REVIEW TABLE CREATED"
    )
    print("=" * 72)

    print(
        f"Total trail names: "
        f"{len(review):,}"
    )

    print(
        f"Auto-excluded: "
        f"{excluded:,}"
    )

    print(
        f"Needs human review: "
        f"{needs_review:,}"
    )

    print()
    print(
        f"Saved to:"
        f"\n  {output_path}"
    )

    print()
    print(
        "Top 80 trails needing review:"
    )

    print()

    print(
        review[
            review[
                "review_status"
            ]
            == ""
        ][
            [
                "trail_name",
                "distance_miles",
                "segment_count",
            ]
        ]
        .head(80)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()