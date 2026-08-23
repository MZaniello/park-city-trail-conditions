from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import random
import time

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

CONDITION_ICONS = {
    "IDEAL": "🟢",
    "GOOD": "🔵",
    "MARGINAL": "🟡",
    "WET": "🟠",
    "POOR": "🔴",
}


# ============================================================
# TRAIL / PREDICTION DATA
# ============================================================


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


def load_predictions(project_root):
    """Load Baseline v2 forecast condition predictions."""

    prediction_path = (
        project_root
        / "data"
        / "processed"
        / "forecast_condition_predictions_v2.csv"
    )

    if not prediction_path.exists():
        return pd.DataFrame()

    predictions = pd.read_csv(
        prediction_path,
        parse_dates=["date"],
    )

    return predictions


def format_condition(condition):
    """Add a visual indicator to condition labels."""

    icon = CONDITION_ICONS.get(
        condition,
        "⚪",
    )

    return f"{icon} {condition}"


def prepare_day(predictions, target_date):
    """Get and rank predictions for one date."""

    target_date = pd.Timestamp(target_date)

    day = predictions[
        predictions["date"].dt.normalize()
        == target_date.normalize()
    ].copy()

    if day.empty:
        return day

    day = day.sort_values(
        [
            "rideability_score",
            "trail_name",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    day["rank"] = day.index + 1

    day["display_condition"] = (
        day["predicted_condition"]
        .apply(format_condition)
    )

    return day


def show_best_bets(day):
    """Display the top three trails for a given day."""

    if day.empty:
        st.info(
            "No prediction data is available for this date."
        )
        return

    st.subheader("🏆 Best Bets")

    top = day.head(3)

    medals = [
        "🥇",
        "🥈",
        "🥉",
    ]

    columns = st.columns(3)

    for index, (_, row) in enumerate(
        top.iterrows()
    ):
        with columns[index]:

            st.markdown(
                f"### {medals[index]} "
                f"{row['trail_name']}"
            )

            st.metric(
                "Rideability",
                f"{int(row['rideability_score'])}/100",
            )

            st.write(
                format_condition(
                    row["predicted_condition"]
                )
            )


def show_rankings(day):
    """Display all trail rankings for a date."""

    if day.empty:
        return

    st.subheader("All Trails")

    display = day[
        [
            "rank",
            "trail_name",
            "rideability_score",
            "display_condition",
            "precip_1d",
            "precip_3d",
            "days_since_precip",
        ]
    ].copy()

    display = display.rename(
        columns={
            "rank": "#",
            "trail_name": "Trail",
            "rideability_score": "Score",
            "display_condition": "Condition",
            "precip_1d": "Rain Today",
            "precip_3d": "Rain 3d",
            "days_since_precip": "Days Dry",
        }
    )

    display["Rain Today"] = (
        display["Rain Today"]
        .map(lambda x: f'{x:.2f}"')
    )

    display["Rain 3d"] = (
        display["Rain 3d"]
        .map(lambda x: f'{x:.2f}"')
    )

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )


def show_day_details(day):
    """Allow inspection of an individual trail prediction."""

    if day.empty:
        return

    st.subheader("Trail Details")

    selected_trail = st.selectbox(
        "View prediction details",
        day["trail_name"].tolist(),
        key=f"details_{day['date'].iloc[0]}",
    )

    trail = day[
        day["trail_name"]
        == selected_trail
    ].iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Rideability",
            f"{int(trail['rideability_score'])}/100",
        )

        st.metric(
            "3-Day Precipitation",
            f'{trail["precip_3d"]:.2f}"',
        )

    with col2:
        st.metric(
            "Condition",
            trail["predicted_condition"],
        )

        st.metric(
            "Days Since Precipitation",
            int(trail["days_since_precip"]),
        )

    st.caption(
        f"Model reasoning: {trail['reason']}"
    )


def show_today_or_tomorrow(
    predictions,
    target_date,
):
    """Display one day's recommendation dashboard."""

    day = prepare_day(
        predictions,
        target_date,
    )

    show_best_bets(day)

    st.divider()

    show_rankings(day)

    st.divider()

    show_day_details(day)


def show_seven_day_summary(predictions):
    """Display top recommendations for every forecast date."""

    if predictions.empty:
        st.info(
            "No forecast predictions are available."
        )
        return

    dates = sorted(
        predictions["date"]
        .dt.normalize()
        .unique()
    )

    for date_value in dates:

        date_value = pd.Timestamp(
            date_value
        )

        day = prepare_day(
            predictions,
            date_value,
        )

        if day.empty:
            continue

        best = day.iloc[0]

        st.markdown(
            f"### {date_value.strftime('%A, %b %d')}"
        )

        st.write(
            f"**Best bet:** "
            f"{best['trail_name']} — "
            f"{int(best['rideability_score'])}/100 "
            f"{format_condition(best['predicted_condition'])}"
        )

        top_five = day.head(5)[
            [
                "trail_name",
                "rideability_score",
                "predicted_condition",
            ]
        ].copy()

        top_five["predicted_condition"] = (
            top_five[
                "predicted_condition"
            ].apply(
                format_condition
            )
        )

        top_five = top_five.rename(
            columns={
                "trail_name": "Trail",
                "rideability_score": "Score",
                "predicted_condition": "Condition",
            }
        )

        st.dataframe(
            top_five,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# GOOGLE SHEETS
# ============================================================


def get_worksheet():
    """
    Connect to Google Sheets.

    Retries temporary Google API failures so a brief 503 does
    not immediately break condition reporting.
    """

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_info(
        dict(
            st.secrets[
                "google_service_account"
            ]
        ),
        scopes=scopes,
    )

    client = gspread.authorize(
        credentials
    )

    max_attempts = 5

    for attempt in range(max_attempts):

        try:
            spreadsheet = client.open_by_key(
                st.secrets[
                    "google_sheet"
                ]["spreadsheet_id"]
            )

            worksheet = spreadsheet.worksheet(
                st.secrets[
                    "google_sheet"
                ]["worksheet_name"]
            )

            return worksheet

        except Exception:

            if attempt == max_attempts - 1:
                raise

            wait_seconds = (
                2 ** attempt
                + random.uniform(0, 1)
            )

            time.sleep(
                wait_seconds
            )


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
        return pd.DataFrame(
            columns=columns
        )

    return pd.DataFrame(
        records
    )


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
        now.isoformat(
            timespec="seconds"
        ),
        now.strftime(
            "%Y-%m-%d"
        ),
        now.strftime(
            "%H:%M"
        ),
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

    if len(values) <= 1:
        return False

    worksheet.delete_rows(
        len(values)
    )

    return True


# ============================================================
# CONDITION REPORT FORM
# ============================================================


def show_report_form(
    trails,
    reports,
):
    """Display the field condition-report form."""

    st.header(
        "📝 Report Trail Conditions"
    )

    st.write(
        "Help improve the model by logging "
        "how a trail is actually riding."
    )

    trail_name = st.selectbox(
        "Trail",
        trails,
        key="report_trail",
    )

    st.subheader(
        "Condition"
    )

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

    source = st.radio(
        "Report source",
        REPORT_SOURCES,
        horizontal=True,
    )

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
                f"Saved: "
                f"{trail_name} — "
                f"{condition}"
            )

            st.rerun()

        except Exception as error:

            st.error(
                "Report could not be saved."
            )

            st.exception(
                error
            )

    reports = load_reports()

    if not reports.empty:

        st.divider()

        st.subheader(
            "Recent Reports"
        )

        st.dataframe(
            reports
            .tail(10)
            .iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )

        st.write(
            f"Total reports collected: "
            f"**{len(reports)}**"
        )

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
                            "There are no reports "
                            "to delete."
                        )

                except Exception as error:

                    st.error(
                        "Could not delete "
                        "the last report."
                    )

                    st.exception(
                        error
                    )


def show_reporting_unavailable():
    """Show a graceful fallback if Google Sheets is unavailable."""

    st.header(
        "📝 Report Trail Conditions"
    )

    st.info(
        "Condition reporting is temporarily unavailable "
        "because the reporting backend could not be reached. "
        "Trail predictions are still available above."
    )


# ============================================================
# MAIN APP
# ============================================================


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    trails = load_trails(
        project_root
    )

    predictions = load_predictions(
        project_root
    )

    st.set_page_config(
        page_title=(
            "Park City Trail Conditions"
        ),
        page_icon="🚵",
        layout="centered",
    )

    # ---------------------------------------------------------
    # GOOGLE SHEETS CONNECTION
    # ---------------------------------------------------------

    reports_available = True

    try:

        reports = load_reports()

    except Exception:

        reports_available = False

        reports = pd.DataFrame()

    # ---------------------------------------------------------
    # HEADER
    # ---------------------------------------------------------

    st.title(
        "🚵 Park City Trail Conditions"
    )

    st.caption(
        "Weather + terrain based trail "
        "condition estimates for Park City, Utah."
    )

    st.warning(
        "These are experimental model estimates, "
        "not official trail-status reports. "
        "Use local closures and posted trail "
        "conditions when available."
    )

    if not reports_available:

        st.warning(
            "Condition reporting is temporarily unavailable, "
            "but trail predictions are still working."
        )

    # ---------------------------------------------------------
    # PREDICTION DASHBOARD
    # ---------------------------------------------------------

    if predictions.empty:

        st.warning(
            "Forecast predictions have not "
            "been generated yet."
        )

    else:

        park_city_now = datetime.now(
            PARK_CITY_TIMEZONE
        )

        today = park_city_now.date()

        tomorrow = (
            today
            + timedelta(days=1)
        )

        tab_today, tab_tomorrow, tab_week = (
            st.tabs(
                [
                    "Today",
                    "Tomorrow",
                    "7-Day",
                ]
            )
        )

        with tab_today:

            st.header(
                today.strftime(
                    "%A, %B %d"
                )
            )

            show_today_or_tomorrow(
                predictions,
                today,
            )

        with tab_tomorrow:

            st.header(
                tomorrow.strftime(
                    "%A, %B %d"
                )
            )

            show_today_or_tomorrow(
                predictions,
                tomorrow,
            )

        with tab_week:

            st.header(
                "7-Day Outlook"
            )

            show_seven_day_summary(
                predictions
            )

    # ---------------------------------------------------------
    # REPORTING SYSTEM
    # ---------------------------------------------------------

    st.divider()

    if reports_available:

        show_report_form(
            trails,
            reports,
        )

    else:

        show_reporting_unavailable()


if __name__ == "__main__":
    main()