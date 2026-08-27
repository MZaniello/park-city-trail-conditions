from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT SCOPE
# ============================================================

IN_SCOPE_AREAS = {
    "Summit Park",
    "Pinebrook",
    "Jeremy Ranch / Glenwild",
    "Park City / PCMR",
    "Deer Valley",
    "Round Valley",
    "Clark Ranch",
    "Jordanelle",
}

OUT_OF_SCOPE_AREAS = {
    "Wasatch Crest / Millcreek",
}


# ============================================================
# ORIGINAL TRUSTED TRAILS
# ============================================================
#
# These are the 19 trails already used successfully throughout
# the existing modeling pipeline.
#

TRUSTED_TRAILS = {
    "9K Trail",
    "Apex",
    "Armstrong",
    "Big Easy",
    "CMG",
    "Cyn City",
    "Empire Link",
    "Flagstaff Loop",
    "Jenni's Trail upper",
    "Keystone Trail",
    "Lost Prospector",
    "Mid-Mountain Trail",
    "Mother Urban",
    "Rambler",
    "Ripple Trail",
    "Round Valley Express",
    "Solamere",
    "Spiro Trail",
    "Tidal Wave",
}


# ============================================================
# OBVIOUS EXCLUSION WORDS
# ============================================================
#
# Conservative list. These do not mean a feature can NEVER be
# ridden by bikes; they mean it is unlikely to be useful as an
# individual MTB trail-condition prediction target.
#

EXCLUDE_KEYWORDS = [
    "service",
    "access",
    "parking",
    "drive",
    "court",
    "highway",
    "ski run",
    "lift",
    "road",
]


# Exceptions to road-like wording.
KEEP_NAME_EXCEPTIONS = {
    "Road to WOS",
}


def contains_exclusion_keyword(name):
    """Check for obvious road/service/access-type names."""

    if name in KEEP_NAME_EXCEPTIONS:
        return False

    lowered = name.lower()

    return any(
        keyword in lowered
        for keyword in EXCLUDE_KEYWORDS
    )


def has_value(value):
    """
    Determine whether an OSM field contains meaningful data.
    Handles blank strings and NaN values.
    """

    if pd.isna(value):
        return False

    value = str(value).strip()

    return value not in {
        "",
        "nan",
        "None",
        "[]",
    }


def main():

    project_root = Path(__file__).resolve().parents[2]

    review_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_review_with_locations.csv"
    )

    candidate_path = (
        project_root
        / "data"
        / "processed"
        / "expanded_trail_candidates.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_triage.csv"
    )

    # ---------------------------------------------------------
    # LOAD REVIEW TABLE
    # ---------------------------------------------------------

    print("Loading master trail review table...")

    review = pd.read_csv(
        review_path,
        keep_default_na=False,
    )

    print(
        f"Trail names: {len(review):,}"
    )

    # ---------------------------------------------------------
    # LOAD SEGMENT-LEVEL OSM METADATA
    # ---------------------------------------------------------

    print(
        "Loading candidate OSM metadata..."
    )

    segments = pd.read_csv(
        candidate_path,
        keep_default_na=False,
    )

    # ---------------------------------------------------------
    # BUILD PER-TRAIL BIKE / MTB SIGNALS
    # ---------------------------------------------------------

    metadata_rows = []

    for trail_name, group in segments.groupby(
        "trail_name"
    ):

        mtb_signal = False
        bicycle_signal = False
        surface_signal = False

        for column in [
            "mtb:scale",
            "mtb:scale:uphill",
        ]:
            if column in group.columns:

                if group[column].apply(
                    has_value
                ).any():

                    mtb_signal = True

        if "bicycle" in group.columns:

            bicycle_values = (
                group["bicycle"]
                .astype(str)
                .str.lower()
                .str.strip()
            )

            bicycle_signal = bicycle_values.isin(
                [
                    "yes",
                    "designated",
                    "permissive",
                ]
            ).any()

        if "surface" in group.columns:

            surface_signal = (
                group["surface"]
                .apply(
                    has_value
                )
                .any()
            )

        metadata_rows.append(
            {
                "trail_name":
                    trail_name,

                "has_mtb_tag":
                    mtb_signal,

                "has_bicycle_tag":
                    bicycle_signal,

                "has_surface_tag":
                    surface_signal,
            }
        )

    metadata = pd.DataFrame(
        metadata_rows
    )

    # ---------------------------------------------------------
    # MERGE
    # ---------------------------------------------------------

    catalog = review.merge(
        metadata,
        on="trail_name",
        how="left",
        validate="one_to_one",
    )

    for column in [
        "has_mtb_tag",
        "has_bicycle_tag",
        "has_surface_tag",
    ]:

        catalog[column] = (
            catalog[column]
            .fillna(False)
            .astype(bool)
        )

    # ---------------------------------------------------------
    # TRIAGE FLAGS
    # ---------------------------------------------------------

    catalog[
        "trusted_existing"
    ] = catalog[
        "trail_name"
    ].isin(
        TRUSTED_TRAILS
    )

    catalog[
        "out_of_scope_area"
    ] = catalog[
        "area_guess"
    ].isin(
        OUT_OF_SCOPE_AREAS
    )

    catalog[
        "obvious_name_exclusion"
    ] = catalog[
        "trail_name"
    ].apply(
        contains_exclusion_keyword
    )

    catalog[
        "too_short"
    ] = (
        pd.to_numeric(
            catalog[
                "distance_miles"
            ],
            errors="coerce",
        )
        < 0.10
    )

    # ---------------------------------------------------------
    # ASSIGN TRIAGE STATUS
    # ---------------------------------------------------------

    catalog[
        "triage_status"
    ] = "REVIEW"

    catalog[
        "triage_reason"
    ] = ""

    # ---------------------------------------------------------
    # AUTO EXCLUDE
    # ---------------------------------------------------------

    out_of_scope_mask = (
        catalog[
            "out_of_scope_area"
        ]
        & ~catalog[
            "trusted_existing"
        ]
    )

    catalog.loc[
        out_of_scope_mask,
        "triage_status",
    ] = "AUTO_EXCLUDE"

    catalog.loc[
        out_of_scope_mask,
        "triage_reason",
    ] = "outside current Park City study area"

    name_exclusion_mask = (
        catalog[
            "obvious_name_exclusion"
        ]
        & ~catalog[
            "trusted_existing"
        ]
    )

    catalog.loc[
        name_exclusion_mask,
        "triage_status",
    ] = "AUTO_EXCLUDE"

    catalog.loc[
        name_exclusion_mask,
        "triage_reason",
    ] = "road/service/access-style name"

    short_mask = (
        catalog[
            "too_short"
        ]
        & ~catalog[
            "trusted_existing"
        ]
    )

    catalog.loc[
        short_mask,
        "triage_status",
    ] = "AUTO_EXCLUDE"

    catalog.loc[
        short_mask,
        "triage_reason",
    ] = "very short mapped fragment"

    # ---------------------------------------------------------
    # AUTO INCLUDE TRUSTED TRAILS
    # ---------------------------------------------------------

    trusted_mask = (
        catalog[
            "trusted_existing"
        ]
    )

    catalog.loc[
        trusted_mask,
        "triage_status",
    ] = "AUTO_INCLUDE"

    catalog.loc[
        trusted_mask,
        "triage_reason",
    ] = "existing validated project trail"

    # ---------------------------------------------------------
    # AUTO INCLUDE STRONG MTB SIGNAL
    # ---------------------------------------------------------
    #
    # Only do this for in-scope names that weren't already
    # excluded.
    #

    mtb_mask = (
        catalog[
            "has_mtb_tag"
        ]
        & catalog[
            "area_guess"
        ].isin(
            IN_SCOPE_AREAS
        )
        & (
            catalog[
                "triage_status"
            ]
            == "REVIEW"
        )
    )

    catalog.loc[
        mtb_mask,
        "triage_status",
    ] = "AUTO_INCLUDE"

    catalog.loc[
        mtb_mask,
        "triage_reason",
    ] = "explicit MTB metadata in OpenStreetMap"

    # ---------------------------------------------------------
    # REVIEW PRIORITY
    # ---------------------------------------------------------
    #
    # Higher priority = review earlier.
    #

    catalog[
        "review_priority"
    ] = 0

    review_mask = (
        catalog[
            "triage_status"
        ]
        == "REVIEW"
    )

    catalog.loc[
        review_mask
        & catalog[
            "has_bicycle_tag"
        ],
        "review_priority",
    ] += 3

    catalog.loc[
        review_mask
        & catalog[
            "has_surface_tag"
        ],
        "review_priority",
    ] += 1

    catalog.loc[
        review_mask
        & (
            pd.to_numeric(
                catalog[
                    "distance_miles"
                ],
                errors="coerce",
            )
            >= 0.5
        ),
        "review_priority",
    ] += 2

    catalog.loc[
        review_mask
        & (
            pd.to_numeric(
                catalog[
                    "distance_miles"
                ],
                errors="coerce",
            )
            >= 1.0
        ),
        "review_priority",
    ] += 1

    # ---------------------------------------------------------
    # MANUAL COLUMNS
    # ---------------------------------------------------------

    catalog[
        "final_status"
    ] = ""

    catalog[
        "final_area"
    ] = ""

    catalog[
        "review_notes"
    ] = ""

    # ---------------------------------------------------------
    # SORT
    # ---------------------------------------------------------

    status_order = {
        "AUTO_INCLUDE": 0,
        "REVIEW": 1,
        "AUTO_EXCLUDE": 2,
    }

    catalog[
        "_status_order"
    ] = catalog[
        "triage_status"
    ].map(
        status_order
    )

    catalog = catalog.sort_values(
        [
            "_status_order",
            "review_priority",
            "area_guess",
            "distance_miles",
            "trail_name",
        ],
        ascending=[
            True,
            False,
            True,
            False,
            True,
        ],
    ).reset_index(
        drop=True
    )

    catalog = catalog.drop(
        columns=[
            "_status_order"
        ]
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    status_counts = (
        catalog[
            "triage_status"
        ]
        .value_counts()
    )

    print()
    print("=" * 72)
    print(
        "MASTER TRAIL TRIAGE COMPLETE"
    )
    print("=" * 72)

    print()

    print(
        status_counts.to_string()
    )

    print()

    review_count = (
        catalog[
            "triage_status"
        ]
        == "REVIEW"
    ).sum()

    print(
        f"Trails still needing human review: "
        f"{review_count:,}"
    )

    print()

    print(
        f"Saved to:"
        f"\n  {output_path}"
    )

    # ---------------------------------------------------------
    # SHOW HIGHEST PRIORITY REVIEW ITEMS
    # ---------------------------------------------------------

    review_rows = catalog[
        catalog[
            "triage_status"
        ]
        == "REVIEW"
    ].copy()

    print()
    print(
        "Top 100 remaining review candidates:"
    )

    print()

    print(
        review_rows[
            [
                "trail_name",
                "area_guess",
                "distance_miles",
                "has_mtb_tag",
                "has_bicycle_tag",
                "has_surface_tag",
                "review_priority",
            ]
        ]
        .head(100)
        .to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()