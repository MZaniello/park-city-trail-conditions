from pathlib import Path

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


EXPECTED_COLUMNS = [
    "date",
    "time",
    "trail_name",
    "condition",
    "source",
    "trail_section",
    "notes",
]


def get_worksheet():
    """Connect to the Google Sheets reports worksheet."""

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


def main():
    project_root = Path(__file__).resolve().parents[2]

    output_path = (
        project_root
        / "data"
        / "raw"
        / "condition_reports.csv"
    )

    print("Connecting to Google Sheets...")

    worksheet = get_worksheet()

    print("Downloading condition reports...")

    records = worksheet.get_all_records()

    if not records:
        reports = pd.DataFrame(
            columns=EXPECTED_COLUMNS
        )

    else:
        reports = pd.DataFrame(records)

    # ---------------------------------------------------------
    # VALIDATE COLUMNS
    # ---------------------------------------------------------

    missing_columns = (
        set(EXPECTED_COLUMNS)
        - set(reports.columns)
    )

    if missing_columns:
        raise ValueError(
            "Google Sheet is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    reports = reports[EXPECTED_COLUMNS].copy()

    # ---------------------------------------------------------
    # NORMALIZE VALUES
    # ---------------------------------------------------------

    if not reports.empty:

        reports["date"] = pd.to_datetime(
            reports["date"],
            errors="coerce",
        )

        reports["trail_name"] = (
            reports["trail_name"]
            .astype(str)
            .str.strip()
        )

        reports["condition"] = (
            reports["condition"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        reports["source"] = (
            reports["source"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        reports["trail_section"] = (
            reports["trail_section"]
            .astype(str)
            .str.strip()
        )

        reports["notes"] = (
            reports["notes"]
            .astype(str)
            .str.strip()
        )

    # ---------------------------------------------------------
    # SAVE LOCAL SNAPSHOT
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print("\nDownload complete!")
    print(f"Reports downloaded: {len(reports):,}")
    print(f"Saved to: {output_path}")

    if not reports.empty:

        print(
            "Unique trails reported: "
            f"{reports['trail_name'].nunique()}"
        )

        valid_dates = reports[
            "date"
        ].dropna()

        if not valid_dates.empty:
            print(
                "First report date: "
                f"{valid_dates.min().date()}"
            )

            print(
                "Last report date: "
                f"{valid_dates.max().date()}"
            )

        print("\nConditions:")

        print(
            reports["condition"]
            .value_counts()
            .to_string()
        )

        print("\nSources:")

        print(
            reports["source"]
            .value_counts()
            .to_string()
        )

    else:
        print(
            "No reports exist yet. "
            "The local snapshot was created with headers only."
        )


if __name__ == "__main__":
    main()