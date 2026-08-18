from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st


ALLOWED_CONDITIONS = [
    "ideal",
    "dry",
    "wet",
    "muddy",
    "snow",
]

REPORT_SOURCES = [
    "customer",
    "staff",
    "personal",
]


def load_trails(project_root):
    """Load approved trail names from the clean catalog."""

    catalog_path = (
        project_root
        / "data"
        / "processed"
        / "clean_trail_catalog.csv"
    )

    catalog = pd.read_csv(catalog_path)

    return sorted(
        catalog["trail_name"]
        .dropna()
        .unique()
        .tolist()
    )


def load_reports(reports_path):
    """Load existing condition reports."""

    columns = [
        "date",
        "time",
        "trail_name",
        "condition",
        "source",
        "trail_section",
        "notes",
    ]

    if not reports_path.exists():
        return pd.DataFrame(columns=columns)

    reports = pd.read_csv(
        reports_path,
        keep_default_na=False,
    )

    return reports


def save_report(
    reports_path,
    trail_name,
    condition,
    source,
    trail_section,
    notes,
):
    """Append one trail-condition report."""

    now = datetime.now()

    new_report = pd.DataFrame(
        [
            {
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M"),
                "trail_name": trail_name,
                "condition": condition,
                "source": source,
                "trail_section": trail_section.strip(),
                "notes": notes.strip(),
            }
        ]
    )

    existing = load_reports(reports_path)

    combined = pd.concat(
        [existing, new_report],
        ignore_index=True,
    )

    reports_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        reports_path,
        index=False,
    )


def undo_last_report(reports_path):
    """Delete the most recently entered report."""

    reports = load_reports(reports_path)

    if reports.empty:
        return False

    reports = reports.iloc[:-1]

    reports.to_csv(
        reports_path,
        index=False,
    )

    return True


def main():
    project_root = Path(__file__).resolve().parents[1]

    reports_path = (
        project_root
        / "data"
        / "observations"
        / "trail_condition_reports.csv"
    )

    trails = load_trails(project_root)

    st.set_page_config(
        page_title="Park City Trail Report",
        page_icon="🚵",
        layout="centered",
    )

    st.title("🚵 Park City Trail Report")

    st.write(
        "Quickly log how a trail is riding."
    )

    # ---------------------------------------------------------
    # TRAIL
    # ---------------------------------------------------------

    trail_name = st.selectbox(
        "Trail",
        trails,
    )

    # ---------------------------------------------------------
    # CONDITION
    # ---------------------------------------------------------

    st.subheader("Condition")

    condition = st.radio(
        "How was the trail riding?",
        ALLOWED_CONDITIONS,
        horizontal=True,
    )

    st.caption(
        "Ideal = tacky / hero dirt · "
        "Dry = firm or dusty · "
        "Wet = noticeable moisture · "
        "Muddy = riding may damage the trail · "
        "Snow = snow or ice affects riding"
    )

    # ---------------------------------------------------------
    # SOURCE
    # ---------------------------------------------------------

    source = st.radio(
        "Report source",
        REPORT_SOURCES,
        horizontal=True,
    )

    # ---------------------------------------------------------
    # OPTIONAL DETAILS
    # ---------------------------------------------------------

    trail_section = st.text_input(
        "Trail section (optional)",
        placeholder="Example: upper, lower, near Armstrong intersection",
    )

    notes = st.text_area(
        "Notes (optional)",
        placeholder=(
            "Example: tacky overall, "
            "one puddle in shaded section"
        ),
    )

    # ---------------------------------------------------------
    # SUBMIT
    # ---------------------------------------------------------

    if st.button(
        "Submit Report",
        type="primary",
        use_container_width=True,
    ):
        save_report(
            reports_path=reports_path,
            trail_name=trail_name,
            condition=condition,
            source=source,
            trail_section=trail_section,
            notes=notes,
        )

        st.success(
            f"Saved: {trail_name} — {condition}"
        )

    # ---------------------------------------------------------
    # RECENT REPORTS
    # ---------------------------------------------------------

    reports = load_reports(reports_path)

    if not reports.empty:
        st.divider()

        st.subheader("Recent Reports")

        st.dataframe(
            reports.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        st.write(
            f"Total reports collected: **{len(reports)}**"
        )

        # -----------------------------------------------------
        # UNDO
        # -----------------------------------------------------

        with st.expander(
            "Entered something by mistake?"
        ):
            st.write(
                "This deletes the most recently "
                "submitted report."
            )

            if st.button(
                "Undo Last Report"
            ):
                if undo_last_report(reports_path):
                    st.success(
                        "Last report deleted."
                    )
                    st.rerun()
                else:
                    st.warning(
                        "There are no reports to delete."
                    )


if __name__ == "__main__":
    main()