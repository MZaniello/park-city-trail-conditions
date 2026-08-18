from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


PARK_CITY_TIMEZONE = ZoneInfo("America/Denver")

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


def get_worksheet():
    """Connect to the Google Sheet using Streamlit secrets."""

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_info(
        dict(st.secrets["google_service_account"]),
        scopes=scopes,
    )

    client = gspread.authorize(credentials)

    spreadsheet = client.open_by_key(
        st.secrets["google_sheet"]["spreadsheet_id"]
    )

    worksheet = spreadsheet.worksheet(
        st.secrets["google_sheet"]["worksheet_name"]
    )

    return worksheet


def load_reports():
    """Load reports currently stored in Google Sheets."""

    worksheet = get_worksheet()

    records = worksheet.get_all_records()

    columns = [
        "timestamp",
        "date",
        "time",
        "trail_name",
        "condition",
        "source",
        "trail_section",
        "notes",
    ]

    if not records:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(records)


def save_report(
    trail_name,
    condition,
    source,
    trail_section,
    notes,
):
    """Append one trail-condition report to Google Sheets."""

    worksheet = get_worksheet()

    now = datetime.now(
        PARK_CITY_TIMEZONE
    )

    row = [
        now.isoformat(timespec="seconds"),
        now.strftime("%Y-%m-%d"),
        now.strftime("%H:%M"),
        trail_name,
        condition,
        source,
        trail_section.strip(),
        notes.strip(),
    ]

    worksheet.append_row(
    row,
    value_input_option="RAW",
)


def undo_last_report():
    """Delete the most recently entered Google Sheet row."""

    worksheet = get_worksheet()

    values = worksheet.get_all_values()

    # Row 1 is the header.
    if len(values) <= 1:
        return False

    worksheet.delete_rows(
        len(values)
    )

    return True


def main():
    project_root = Path(__file__).resolve().parents[1]

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
    # TEST GOOGLE SHEETS CONNECTION
    # ---------------------------------------------------------

    try:
        reports = load_reports()

    except Exception as error:
        st.error(
            "Could not connect to Google Sheets."
        )

        st.exception(error)

        st.stop()

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
        placeholder=(
            "Example: upper, lower, "
            "near Armstrong intersection"
        ),
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
        try:
            save_report(
                trail_name=trail_name,
                condition=condition,
                source=source,
                trail_section=trail_section,
                notes=notes,
            )

            st.success(
                f"Saved: {trail_name} — {condition}"
            )

            st.rerun()

        except Exception as error:
            st.error(
                "Report could not be saved."
            )
            st.exception(error)

    # ---------------------------------------------------------
    # RELOAD REPORTS AFTER SUBMISSION
    # ---------------------------------------------------------

    reports = load_reports()

    # ---------------------------------------------------------
    # RECENT REPORTS
    # ---------------------------------------------------------

    if not reports.empty:
        st.divider()

        st.subheader("Recent Reports")

        st.dataframe(
            reports.tail(10).iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        st.write(
            f"Total reports collected: "
            f"**{len(reports)}**"
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
                try:
                    if undo_last_report():
                        st.success(
                            "Last report deleted."
                        )
                        st.rerun()

                    else:
                        st.warning(
                            "There are no reports to delete."
                        )

                except Exception as error:
                    st.error(
                        "Could not delete the last report."
                    )
                    st.exception(error)


if __name__ == "__main__":
    main()
