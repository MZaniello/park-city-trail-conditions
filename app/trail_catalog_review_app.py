from pathlib import Path

import pandas as pd
import streamlit as st


AREA_OPTIONS = [
    "Summit Park",
    "Pinebrook",
    "Jeremy Ranch / Glenwild",
    "Park City / PCMR",
    "Deer Valley",
    "Round Valley",
    "Clark Ranch",
    "Jordanelle",
    "Wasatch Crest / Millcreek",
    "Other",
]


def get_paths():
    project_root = Path(__file__).resolve().parents[1]

    input_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_triage.csv"
    )

    progress_path = (
        project_root
        / "data"
        / "processed"
        / "trail_catalog_review_progress.csv"
    )

    final_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_catalog.csv"
    )

    return input_path, progress_path, final_path


def load_catalog():
    """
    Load triage data and restore previous review progress
    if it exists.
    """

    input_path, progress_path, _ = get_paths()

    catalog = pd.read_csv(
        input_path,
        keep_default_na=False,
    )

    # Make sure editable columns exist.
    for column in [
        "final_status",
        "final_area",
        "review_notes",
    ]:
        if column not in catalog.columns:
            catalog[column] = ""

    # ---------------------------------------------------------
    # PRELOAD AUTOMATIC DECISIONS
    # ---------------------------------------------------------

    include_mask = (
        (catalog["triage_status"] == "AUTO_INCLUDE")
        & (catalog["final_status"] == "")
    )

    catalog.loc[
        include_mask,
        "final_status",
    ] = "INCLUDE"

    exclude_mask = (
        (catalog["triage_status"] == "AUTO_EXCLUDE")
        & (catalog["final_status"] == "")
    )

    catalog.loc[
        exclude_mask,
        "final_status",
    ] = "EXCLUDE"

    # Use area guess as the initial editable area.
    empty_area = (
        catalog["final_area"] == ""
    )

    catalog.loc[
        empty_area,
        "final_area",
    ] = catalog.loc[
        empty_area,
        "area_guess",
    ]

    # ---------------------------------------------------------
    # RESTORE SAVED PROGRESS
    # ---------------------------------------------------------

    if progress_path.exists():

        progress = pd.read_csv(
            progress_path,
            keep_default_na=False,
        )

        if not progress.empty:

            saved_columns = [
                "trail_name",
                "final_status",
                "final_area",
                "review_notes",
            ]

            progress = progress[
                [
                    column
                    for column in saved_columns
                    if column in progress.columns
                ]
            ]

            catalog = catalog.merge(
                progress,
                on="trail_name",
                how="left",
                suffixes=("", "_saved"),
                validate="one_to_one",
            )

            for column in [
                "final_status",
                "final_area",
                "review_notes",
            ]:

                saved_column = (
                    f"{column}_saved"
                )

                if saved_column in catalog.columns:

                    saved_values = (
                        catalog[
                            saved_column
                        ]
                        .astype(str)
                    )

                    has_saved_value = (
                        saved_values
                        .str.strip()
                        != ""
                    )

                    catalog.loc[
                        has_saved_value,
                        column,
                    ] = catalog.loc[
                        has_saved_value,
                        saved_column,
                    ]

                    catalog = catalog.drop(
                        columns=[
                            saved_column
                        ]
                    )

    return catalog


def save_progress(catalog):
    """
    Save review state immediately.
    """

    _, progress_path, _ = get_paths()

    progress_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    catalog[
        [
            "trail_name",
            "final_status",
            "final_area",
            "review_notes",
        ]
    ].to_csv(
        progress_path,
        index=False,
    )


def build_final_catalog(catalog):
    """
    Create the final catalog containing included trails only.
    """

    _, _, final_path = get_paths()

    included = catalog[
        catalog["final_status"]
        == "INCLUDE"
    ].copy()

    useful_columns = [
        "trail_name",
        "final_area",
        "latitude",
        "longitude",
        "distance_miles",
        "segment_count",
        "node_count",
        "source",
        "review_notes",
    ]

    useful_columns = [
        column
        for column in useful_columns
        if column in included.columns
    ]

    included = included[
        useful_columns
    ].copy()

    included = included.sort_values(
        [
            "final_area",
            "trail_name",
        ]
    ).reset_index(
        drop=True
    )

    final_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    included.to_csv(
        final_path,
        index=False,
    )

    return final_path, len(included)


def get_review_candidates(catalog):
    """
    Return trails that require a human decision.
    """

    review = catalog[
        catalog["triage_status"]
        == "REVIEW"
    ].copy()

    # Longer and higher-priority trails first.
    review = review.sort_values(
        [
            "review_priority",
            "distance_miles",
            "trail_name",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    )

    return review


def initialize_session(catalog):
    """
    Initialize app state.
    """

    if "catalog" not in st.session_state:
        st.session_state.catalog = catalog

    if "current_index" not in st.session_state:
        st.session_state.current_index = 0

    if "history" not in st.session_state:
        st.session_state.history = []


def next_unreviewed_index(review):
    """
    Find the first trail without a human decision.
    """

    statuses = (
        review["final_status"]
        .astype(str)
        .str.strip()
    )

    unreviewed = review[
        statuses == ""
    ]

    if unreviewed.empty:
        return None

    return review.index.get_loc(
        unreviewed.index[0]
    )


def record_decision(
    trail_name,
    status,
    area,
    notes,
):
    """
    Save a human review decision.
    """

    catalog = st.session_state.catalog

    mask = (
        catalog["trail_name"]
        == trail_name
    )

    old_status = catalog.loc[
        mask,
        "final_status",
    ].iloc[0]

    old_area = catalog.loc[
        mask,
        "final_area",
    ].iloc[0]

    old_notes = catalog.loc[
        mask,
        "review_notes",
    ].iloc[0]

    st.session_state.history.append(
        {
            "trail_name":
                trail_name,
            "final_status":
                old_status,
            "final_area":
                old_area,
            "review_notes":
                old_notes,
        }
    )

    catalog.loc[
        mask,
        "final_status",
    ] = status

    catalog.loc[
        mask,
        "final_area",
    ] = area

    catalog.loc[
        mask,
        "review_notes",
    ] = notes.strip()

    save_progress(
        catalog
    )


def undo_last():
    """
    Undo the most recent decision made in this session.
    """

    if not st.session_state.history:
        return False

    previous = (
        st.session_state.history.pop()
    )

    catalog = (
        st.session_state.catalog
    )

    mask = (
        catalog["trail_name"]
        == previous["trail_name"]
    )

    catalog.loc[
        mask,
        "final_status",
    ] = previous[
        "final_status"
    ]

    catalog.loc[
        mask,
        "final_area",
    ] = previous[
        "final_area"
    ]

    catalog.loc[
        mask,
        "review_notes",
    ] = previous[
        "review_notes"
    ]

    save_progress(
        catalog
    )

    return True


def main():

    st.set_page_config(
        page_title="Trail Catalog Review",
        page_icon="🚵",
        layout="centered",
    )

    catalog = load_catalog()

    initialize_session(
        catalog
    )

    catalog = (
        st.session_state.catalog
    )

    review = get_review_candidates(
        catalog
    )

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    st.title(
        "🚵 Trail Catalog Review"
    )

    st.caption(
        "Review ambiguous OSM trails for the "
        "Park City trail-condition model."
    )

    # ---------------------------------------------------------
    # PROGRESS
    # ---------------------------------------------------------

    total_review = len(
        review
    )

    reviewed_mask = (
        review["final_status"]
        .astype(str)
        .str.strip()
        != ""
    )

    reviewed_count = int(
        reviewed_mask.sum()
    )

    remaining = (
        total_review
        - reviewed_count
    )

    progress = (
        reviewed_count
        / total_review
        if total_review
        else 1.0
    )

    st.progress(
        progress
    )

    col1, col2, col3 = (
        st.columns(3)
    )

    col1.metric(
        "Reviewed",
        reviewed_count,
    )

    col2.metric(
        "Remaining",
        remaining,
    )

    included_total = (
        catalog[
            "final_status"
        ]
        == "INCLUDE"
    ).sum()

    col3.metric(
        "Included trails",
        int(
            included_total
        ),
    )

    # ---------------------------------------------------------
    # COMPLETE
    # ---------------------------------------------------------

    if remaining == 0:

        st.success(
            "All ambiguous trails have been reviewed."
        )

        final_path, trail_count = (
            build_final_catalog(
                catalog
            )
        )

        st.write(
            f"Final catalog contains "
            f"**{trail_count} trails**."
        )

        st.code(
            str(
                final_path
            )
        )

        if st.button(
            "Undo Last Decision"
        ):

            if undo_last():
                st.rerun()

        return

    # ---------------------------------------------------------
    # SELECT NEXT UNREVIEWED TRAIL
    # ---------------------------------------------------------

    unreviewed = review[
        review["final_status"]
        .astype(str)
        .str.strip()
        == ""
    ].copy()

    current = (
        unreviewed.iloc[0]
    )

    trail_name = (
        current["trail_name"]
    )

    # ---------------------------------------------------------
    # TRAIL CARD
    # ---------------------------------------------------------

    st.divider()

    st.subheader(
        trail_name
    )

    area_guess = str(
        current.get(
            "area_guess",
            "",
        )
    )

    distance = pd.to_numeric(
        current.get(
            "distance_miles",
            0,
        ),
        errors="coerce",
    )

    if pd.isna(distance):
        distance = 0

    info1, info2 = (
        st.columns(2)
    )

    info1.metric(
        "Length",
        f"{distance:.2f} mi",
    )

    info2.metric(
        "Suggested area",
        area_guess,
    )

    # ---------------------------------------------------------
    # OSM INFORMATION
    # ---------------------------------------------------------

    with st.expander(
        "OSM information"
    ):

        st.write(
            f"**MTB tag:** "
            f"{current.get('has_mtb_tag', False)}"
        )

        st.write(
            f"**Bicycle tag:** "
            f"{current.get('has_bicycle_tag', False)}"
        )

        st.write(
            f"**Surface tag:** "
            f"{current.get('has_surface_tag', False)}"
        )

        st.write(
            f"**Review priority:** "
            f"{current.get('review_priority', 0)}"
        )

        latitude = current.get(
            "latitude",
            ""
        )

        longitude = current.get(
            "longitude",
            ""
        )

        st.write(
            f"**Location:** "
            f"{latitude}, {longitude}"
        )

    # ---------------------------------------------------------
    # AREA
    # ---------------------------------------------------------

    current_area = str(
        current.get(
            "final_area",
            "",
        )
    )

    if (
        not current_area
        or current_area
        not in AREA_OPTIONS
    ):
        current_area = (
            area_guess
            if area_guess
            in AREA_OPTIONS
            else "Other"
        )

    area_index = (
        AREA_OPTIONS.index(
            current_area
        )
    )

    selected_area = st.selectbox(
        "Area",
        AREA_OPTIONS,
        index=area_index,
        key=f"area_{trail_name}",
    )

    # ---------------------------------------------------------
    # NOTES
    # ---------------------------------------------------------

    notes = st.text_input(
        "Notes (optional)",
        value=str(
            current.get(
                "review_notes",
                "",
            )
        ),
        placeholder=(
            "Example: legitimate Glenwild singletrack"
        ),
        key=f"notes_{trail_name}",
    )

    # ---------------------------------------------------------
    # DECISION BUTTONS
    # ---------------------------------------------------------

    st.write(
        "**Should this trail receive its own "
        "condition prediction?**"
    )

    include_col, exclude_col = (
        st.columns(2)
    )

    with include_col:

        if st.button(
            "✅ Include + Next",
            type="primary",
            use_container_width=True,
        ):

            record_decision(
                trail_name,
                "INCLUDE",
                selected_area,
                notes,
            )

            st.rerun()

    with exclude_col:

        if st.button(
            "❌ Exclude + Next",
            use_container_width=True,
        ):

            record_decision(
                trail_name,
                "EXCLUDE",
                selected_area,
                notes,
            )

            st.rerun()

    # ---------------------------------------------------------
    # SKIP / UNDO
    # ---------------------------------------------------------

    skip_col, undo_col = (
        st.columns(2)
    )

    with skip_col:

        if st.button(
            "⏭ Skip for Now",
            use_container_width=True,
        ):

            # Move this trail temporarily to the end
            # by assigning SKIP.
            record_decision(
                trail_name,
                "SKIP",
                selected_area,
                notes,
            )

            st.rerun()

    with undo_col:

        if st.button(
            "↩ Undo Last",
            use_container_width=True,
        ):

            if undo_last():
                st.rerun()

            else:
                st.warning(
                    "Nothing to undo in this session."
                )

    # ---------------------------------------------------------
    # REVIEWED THIS SESSION
    # ---------------------------------------------------------

    if st.session_state.history:

        st.divider()

        st.caption(
            f"Decisions this session: "
            f"{len(st.session_state.history)}"
        )

    # ---------------------------------------------------------
    # SAVE FINAL CATALOG AT ANY TIME
    # ---------------------------------------------------------

    with st.expander(
        "Catalog tools"
    ):

        if st.button(
            "Build Current Master Catalog"
        ):

            final_path, trail_count = (
                build_final_catalog(
                    catalog
                )
            )

            st.success(
                f"Saved catalog with "
                f"{trail_count} included trails."
            )

            st.code(
                str(
                    final_path
                )
            )


if __name__ == "__main__":
    main()