from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import gspread
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials


# ============================================================
# APP SETTINGS
# ============================================================

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

SURFACE_HELP = {
    "IDEAL": "Tacky / hero-dirt conditions",
    "DRY": "Firm and generally good",
    "DUSTY": "Loose or dusty but still rideable",
    "DAMP": "Moist and potentially soft",
    "WET": "Wet enough to reduce traction",
    "MUDDY": "Likely muddy or damaging to ride",
    "SNOW": "Snow or ice affects riding",
}


# ============================================================
# PATHS
# ============================================================

def get_project_root():
    return Path(__file__).resolve().parents[1]


def get_data_paths():
    root = get_project_root()

    prediction_path = (
        root
        / "data"
        / "processed"
        / "master_forecast_condition_predictions_v3_2.csv"
    )

    raw_forecast_path = (
        root
        / "data"
        / "raw"
        / "master_trail_forecast.csv"
    )

    topography_path = (
        root
        / "data"
        / "processed"
        / "master_trail_topography_features.csv"
    )

    return (
        prediction_path,
        raw_forecast_path,
        topography_path,
    )


# ============================================================
# WEATHER HELPERS
# ============================================================

def weather_code_info(
    code,
    precipitation=0.0,
):
    try:
        code = int(code)
    except Exception:
        return "🌡️", "Forecast"

    try:
        precipitation = float(
            precipitation
        )
    except Exception:
        precipitation = 0.0

    # Open-Meteo can classify a day as drizzle even when the daily
    # accumulation is tiny. For a rider-facing forecast, avoid making
    # every trace-precipitation day look rainy.
    if precipitation < 0.03:
        if code in {51, 53, 55, 56, 57, 61, 63, 65, 80, 81, 82}:
            return "⛅", "Mostly dry"

    if precipitation < 0.08:
        if code in {51, 53, 55, 61, 63, 80, 81}:
            return "🌦️", "Chance of rain"

    mapping = {
        0: ("☀️", "Clear"),
        1: ("🌤️", "Mostly clear"),
        2: ("⛅", "Partly cloudy"),
        3: ("☁️", "Overcast"),
        45: ("🌫️", "Fog"),
        48: ("🌫️", "Fog"),
        51: ("🌦️", "Light drizzle"),
        53: ("🌦️", "Drizzle"),
        55: ("🌧️", "Heavy drizzle"),
        56: ("🌧️", "Freezing drizzle"),
        57: ("🌧️", "Freezing drizzle"),
        61: ("🌦️", "Light rain"),
        63: ("🌧️", "Rain"),
        65: ("🌧️", "Heavy rain"),
        66: ("🌧️", "Freezing rain"),
        67: ("🌧️", "Freezing rain"),
        71: ("🌨️", "Light snow"),
        73: ("🌨️", "Snow"),
        75: ("❄️", "Heavy snow"),
        77: ("❄️", "Snow grains"),
        80: ("🌦️", "Rain showers"),
        81: ("🌧️", "Rain showers"),
        82: ("⛈️", "Heavy showers"),
        85: ("🌨️", "Snow showers"),
        86: ("🌨️", "Snow showers"),
        95: ("⛈️", "Thunderstorms"),
        96: ("⛈️", "Thunderstorms"),
        99: ("⛈️", "Thunderstorms"),
    }

    return mapping.get(
        code,
        ("🌡️", "Forecast"),
    )


# ============================================================
# STYLING
# ============================================================

def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --page: #0b1013;
            --panel: rgba(13, 21, 25, .89);
            --panel-soft: rgba(15, 24, 29, .82);
            --border: rgba(201, 228, 215, .16);
            --text: #f4f7f5;
            --soft: rgba(244,247,245,.66);
            --faint: rgba(244,247,245,.46);
            --green: #67cf73;
            --green-soft: rgba(103,207,115,.14);
            --yellow: #d7c84a;
            --orange: #e28b52;
            --red: #e3675a;
        }

        html, body {
            background: var(--page);
        }

        [data-testid="stHeader"] {
            height: 0 !important;
            min-height: 0 !important;
            background: transparent !important;
        }

        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            display: none !important;
        }

        /*
        Continuous fixed topo field.
        The SVG is sized to the viewport rather than repeated, so the
        contour lines do not visibly cut off and restart mid-page.
        */
        [data-testid="stAppViewContainer"] {
            background-color: #0b1013 !important;
            background-image:
                linear-gradient(
                    90deg,
                    rgba(7,12,15,.88) 0%,
                    rgba(7,12,15,.68) 34%,
                    rgba(7,12,15,.72) 68%,
                    rgba(7,12,15,.86) 100%
                ),
                url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1600 1000' preserveAspectRatio='xMidYMid slice'%3E%3Cg fill='none' stroke='%2379b66d' stroke-opacity='.24' stroke-width='1.25'%3E%3Cpath d='M-100 90 C110 5 270 180 455 91 S780 -15 1010 118 S1375 204 1700 82'/%3E%3Cpath d='M-120 130 C100 38 280 220 472 129 S797 18 1028 151 S1390 239 1725 119'/%3E%3Cpath d='M-130 173 C92 74 286 259 489 169 S815 53 1045 189 S1405 278 1730 158'/%3E%3Cpath d='M-145 219 C86 113 295 301 507 212 S836 93 1061 231 S1421 323 1744 200'/%3E%3Cpath d='M-155 269 C82 158 307 346 529 258 S858 139 1079 275 S1439 369 1758 245'/%3E%3Cpath d='M-165 322 C86 206 326 390 554 307 S884 191 1101 324 S1464 416 1775 293'/%3E%3Cpath d='M-150 412 C44 300 232 450 409 387 S705 269 935 402 S1268 504 1608 390'/%3E%3Cpath d='M-165 458 C38 342 241 497 424 430 S727 311 958 445 S1295 549 1638 432'/%3E%3Cpath d='M-175 507 C41 388 255 542 445 477 S751 357 981 490 S1321 594 1662 477'/%3E%3Cpath d='M55 690 C191 596 344 701 455 650 S676 567 862 687 S1139 805 1399 704 S1613 620 1700 674'/%3E%3Cpath d='M30 738 C176 638 349 746 470 694 S700 608 888 729 S1168 850 1424 749 S1630 664 1715 718'/%3E%3Cpath d='M8 788 C163 681 357 793 487 740 S725 650 914 772 S1196 895 1451 794 S1650 708 1733 764'/%3E%3Cpath d='M-12 840 C156 727 368 841 505 788 S751 694 942 817 S1224 939 1478 840 S1668 753 1750 809'/%3E%3C/g%3E%3C/svg%3E");
            background-size: cover, 100vw 100vh;
            background-repeat: no-repeat, no-repeat;
            background-position: center center, center center;
            background-attachment: fixed, fixed;
        }

        [data-testid="stMainBlockContainer"] {
            max-width: 1320px !important;
            padding-top: 2.8rem !important;
            padding-bottom: 4rem !important;
        }

        [data-testid="stSidebar"] {
            background: rgba(7,14,18,.94) !important;
            border-right: 1px solid rgba(255,255,255,.10);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 2.0rem;
        }

        .brand {
            margin-bottom: 1.7rem;
        }

        .brand-main {
            font-size: 1.04rem;
            font-weight: 860;
            letter-spacing: .055em;
        }

        .brand-sub {
            color: var(--soft);
            font-size: .72rem;
            letter-spacing: .10em;
            text-transform: uppercase;
            margin-top: .12rem;
        }

        .brand-mark {
            color: var(--green);
            font-size: 1.2rem;
            margin-right: .35rem;
        }

        .nav-label {
            color: var(--faint);
            font-size: .68rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: .08em;
            margin: .9rem 0 .35rem 0;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            gap: .2rem !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            padding: .56rem .65rem !important;
            border-radius: 10px !important;
            border: 1px solid transparent !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            background: rgba(255,255,255,.045);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
            display: none !important;
        }

        [data-baseweb="select"] > div {
            background: rgba(5,10,13,.78) !important;
            border: 1px solid rgba(255,255,255,.10) !important;
            border-radius: 11px !important;
        }

        .zone-kicker {
            color: var(--green);
            font-size: .78rem;
            font-weight: 800;
            letter-spacing: .01em;
            margin-bottom: .15rem;
        }

        .trail-title {
            font-size: clamp(2.65rem, 5.2vw, 4.7rem);
            font-weight: 880;
            line-height: .96;
            letter-spacing: -.055em;
            margin: 0 0 .38rem 0;
        }

        .trail-date {
            color: var(--soft);
            font-size: .9rem;
            margin-bottom: 1rem;
        }

        .glass-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 18px;
            box-shadow: 0 18px 50px rgba(0,0,0,.24);
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
        }

        .score-card {
            padding: 1.15rem 1.25rem 1rem 1.25rem;
            margin-bottom: .85rem;
        }

        .score-grid {
            display: grid;
            grid-template-columns: 145px 1fr;
            gap: 1.1rem;
            align-items: center;
        }

        .score-ring {
            width: 138px;
            height: 138px;
            border-radius: 50%;
            border: 10px solid rgba(103,207,115,.70);
            box-shadow:
                inset 0 0 32px rgba(103,207,115,.10),
                0 0 34px rgba(103,207,115,.08);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            background: rgba(0,0,0,.16);
        }

        .score-number {
            font-size: 3.1rem;
            font-weight: 880;
            line-height: .92;
            letter-spacing: -.06em;
        }

        .score-denom {
            color: var(--soft);
            font-weight: 720;
            font-size: .84rem;
            margin-top: .15rem;
        }

        .tag {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border-radius: 999px;
            padding: .31rem .62rem;
            font-size: .68rem;
            font-weight: 840;
            letter-spacing: .04em;
            white-space: nowrap;
            border: 1px solid transparent;
        }

        .excellent { background: rgba(62,161,78,.20); color: #77da86; border-color: rgba(62,161,78,.30); }
        .good      { background: rgba(130,169,52,.18); color: #b3d65a; border-color: rgba(130,169,52,.28); }
        .fair      { background: rgba(205,165,48,.18); color: #e1bf55; border-color: rgba(205,165,48,.28); }
        .poor      { background: rgba(215,112,58,.18); color: #ee9869; border-color: rgba(215,112,58,.30); }
        .avoid     { background: rgba(198,71,67,.20); color: #ed8e89; border-color: rgba(198,71,67,.30); }

        .ideal { background: rgba(62,161,78,.18); color: #80da8e; border-color: rgba(62,161,78,.28); }
        .dry   { background: rgba(183,151,73,.17); color: #dbbf7a; border-color: rgba(183,151,73,.28); }
        .dusty { background: rgba(194,137,60,.18); color: #e2b26f; border-color: rgba(194,137,60,.30); }
        .damp  { background: rgba(74,130,172,.20); color: #91c1df; border-color: rgba(74,130,172,.30); }
        .wet   { background: rgba(49,102,159,.22); color: #8ab6e4; border-color: rgba(49,102,159,.30); }
        .muddy { background: rgba(126,80,53,.25); color: #d2a083; border-color: rgba(126,80,53,.34); }
        .snow  { background: rgba(173,205,223,.18); color: #d0e7f3; border-color: rgba(173,205,223,.30); }

        .score-label {
            color: var(--soft);
            font-size: .83rem;
            margin-top: .4rem;
        }

        .score-summary {
            border-top: 1px solid rgba(255,255,255,.08);
            margin-top: .8rem;
            padding-top: .72rem;
            color: var(--soft);
            font-size: .84rem;
            line-height: 1.45;
        }

        .score-summary strong {
            color: var(--text);
        }

        .section-title {
            font-size: 1.13rem;
            font-weight: 830;
            margin: 1rem 0 .65rem 0;
            letter-spacing: -.02em;
        }

        .forecast-strip {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: .35rem;
        }

        .ride-day {
            background: rgba(9,16,20,.75);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 10px;
            padding: .55rem .22rem;
            text-align: center;
            min-width: 0;
        }

        .ride-day-name {
            color: var(--soft);
            font-size: .60rem;
            font-weight: 780;
            text-transform: uppercase;
        }

        .ride-score {
            font-size: 1.3rem;
            font-weight: 860;
            margin: .15rem 0;
        }

        .ride-rating {
            font-size: .53rem;
            font-weight: 840;
            white-space: nowrap;
        }

        .weather-card {
            padding: 1rem 1rem .85rem 1rem;
            margin-bottom: .85rem;
        }

        .weather-strip {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: .25rem;
        }

        .weather-day {
            text-align: center;
            border-right: 1px solid rgba(255,255,255,.07);
            padding: .35rem .15rem;
            min-width: 0;
        }

        .weather-day:last-child {
            border-right: 0;
        }

        .weather-day-name {
            font-size: .58rem;
            color: var(--soft);
            text-transform: uppercase;
            font-weight: 760;
        }

        .weather-icon {
            font-size: 1.55rem;
            margin: .18rem 0 .08rem 0;
        }

        .weather-high {
            font-size: 1.05rem;
            font-weight: 840;
        }

        .weather-low {
            color: var(--soft);
            font-size: .72rem;
        }

        .weather-rain {
            color: #82c9ed;
            font-size: .58rem;
            margin-top: .1rem;
        }

        .weather-name {
            color: var(--faint);
            font-size: .52rem;
            margin-top: .08rem;
            overflow: hidden;
            white-space: nowrap;
            text-overflow: ellipsis;
        }

        .weather-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: .4rem;
            margin-top: .65rem;
        }

        .weather-stat {
            background: rgba(7,13,17,.64);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 9px;
            padding: .45rem .4rem;
            text-align: center;
        }

        .weather-stat-label {
            color: var(--faint);
            font-size: .56rem;
            text-transform: uppercase;
        }

        .weather-stat-value {
            font-size: .82rem;
            font-weight: 800;
            margin-top: .08rem;
        }

        .stats-strip {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            margin-top: .8rem;
        }

        .stat-cell {
            padding: .78rem .7rem;
            border-right: 1px solid rgba(255,255,255,.08);
            text-align: center;
        }

        .stat-cell:last-child {
            border-right: 0;
        }

        .stat-icon {
            font-size: 1rem;
        }

        .stat-label {
            color: var(--faint);
            font-size: .58rem;
            text-transform: uppercase;
            margin-top: .1rem;
        }

        .stat-value {
            font-size: .83rem;
            font-weight: 800;
            margin-top: .15rem;
        }

        .sun-card {
            padding: .72rem .85rem;
            margin-top: .6rem;
        }

        .sun-title {
            font-size: .82rem;
            font-weight: 810;
        }

        .sun-copy {
            color: var(--soft);
            font-size: .76rem;
            line-height: 1.4;
            margin-top: .18rem;
        }

        .report-preview-card {
            padding: .65rem .72rem;
            margin-bottom: .4rem;
        }

        .report-preview-title {
            font-size: .78rem;
            font-weight: 810;
        }

        .report-preview-meta {
            color: var(--faint);
            font-size: .64rem;
            margin-top: .12rem;
        }

        .report-preview-copy {
            color: var(--soft);
            font-size: .71rem;
            line-height: 1.35;
            margin-top: .18rem;
        }

        .rank-button div.stButton > button {
            width: 100%;
            min-height: 44px;
            justify-content: flex-start;
            text-align: left;
            background: rgba(9,16,20,.76);
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 10px;
            font-size: .74rem;
            font-weight: 740;
        }

        .report-page-title {
            font-size: clamp(2.4rem, 5vw, 4.2rem);
            font-weight: 870;
            letter-spacing: -.05em;
            margin: 0 0 .25rem 0;
        }

        .report-card {
            padding: .85rem .95rem;
            margin-bottom: .55rem;
        }

        .report-title {
            font-size: .88rem;
            font-weight: 820;
        }

        .report-meta {
            color: var(--faint);
            font-size: .67rem;
            margin-top: .2rem;
        }

        .report-copy {
            color: var(--soft);
            font-size: .76rem;
            line-height: 1.4;
            margin-top: .22rem;
        }

        .stButton > button {
            border-radius: 10px !important;
        }

        div.stButton > button[kind="primary"] {
            background: var(--green) !important;
            color: #071008 !important;
            border-color: var(--green) !important;
            font-weight: 820 !important;
        }

        /*
        Kill browser caret / text selection everywhere except real inputs.
        */
        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stSidebar"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stMetric"],
        .stMarkdown,
        .element-container,
        .glass-card,
        .trail-title,
        .zone-kicker,
        .trail-date,
        .brand {
            caret-color: transparent !important;
            user-select: none !important;
            -webkit-user-select: none !important;
        }

        input,
        textarea,
        [contenteditable="true"],
        [role="textbox"] {
            caret-color: auto !important;
            user-select: text !important;
            -webkit-user-select: text !important;
        }

        button,
        [role="button"],
        [data-baseweb="select"] {
            cursor: pointer !important;
        }

        [data-testid="stImage"] img {
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,.08);
        }

        @media (max-width: 900px) {
            [data-testid="stMainBlockContainer"] {
                padding-left: .75rem !important;
                padding-right: .75rem !important;
            }

            .score-grid {
                grid-template-columns: 112px 1fr;
            }

            .score-ring {
                width: 104px;
                height: 104px;
                border-width: 8px;
            }

            .score-number {
                font-size: 2.5rem;
            }

            .weather-strip,
            .forecast-strip {
                grid-template-columns: repeat(4, minmax(0, 1fr));
            }

            .weather-day {
                border-right: 0;
            }
        }

        /* ================= V9 weather + navigation fixes ================= */

        /* Keep Streamlit's sidebar restore control accessible even though
           the normal header chrome is visually hidden. */
        [data-testid="stSidebarCollapsedControl"] {
            display: flex !important;
            position: fixed !important;
            top: .75rem !important;
            left: .75rem !important;
            z-index: 999999 !important;
        }

        [data-testid="stSidebarCollapsedControl"] button {
            width: 2.45rem !important;
            height: 2.45rem !important;
            border-radius: 12px !important;
            background: rgba(10,17,21,.90) !important;
            border: 1px solid rgba(255,255,255,.14) !important;
            color: #f4f7f5 !important;
            box-shadow: 0 8px 24px rgba(0,0,0,.28) !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            cursor: pointer !important;
        }

        [data-testid="stSidebarCollapseButton"] button {
            border-radius: 10px !important;
        }

        /* Streamlit container around the weather module. */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 18px !important;
        }

        .weather-native-card {
            min-height: 150px;
            padding: .62rem .18rem .55rem .18rem;
            text-align: center;
            border-radius: 11px;
            background: rgba(7,14,18,.56);
            border: 1px solid rgba(255,255,255,.065);
            overflow: hidden;
        }

        .weather-native-card .weather-day-name {
            font-size: .60rem;
            margin-bottom: .12rem;
        }

        .weather-native-card .weather-icon {
            font-size: 1.6rem;
            line-height: 1.2;
            min-height: 2rem;
            margin: .12rem 0 .12rem 0;
        }

        .weather-native-card .weather-high {
            font-size: 1.02rem;
            line-height: 1.1;
        }

        .weather-native-card .weather-low {
            font-size: .70rem;
            margin-top: .1rem;
        }

        .weather-native-card .weather-rain {
            font-size: .55rem;
            margin-top: .24rem;
            white-space: nowrap;
        }

        .weather-native-card .weather-name {
            font-size: .50rem;
            line-height: 1.15;
            margin-top: .15rem;
            white-space: normal;
            overflow: visible;
            text-overflow: unset;
        }

        .weather-stat-native {
            background: rgba(7,14,18,.56);
            border: 1px solid rgba(255,255,255,.065);
            border-radius: 10px;
            padding: .48rem .30rem;
            text-align: center;
            min-height: 60px;
        }

        /* Make weather columns gracefully compress instead of exploding
           vertically or leaving a huge empty region. */
        div[data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }

        @media (max-width: 1050px) {
            .weather-native-card {
                min-height: 138px;
                padding-left: .08rem;
                padding-right: .08rem;
            }

            .weather-native-card .weather-icon {
                font-size: 1.35rem;
            }

            .weather-native-card .weather-high {
                font-size: .92rem;
            }

            .weather-native-card .weather-rain,
            .weather-native-card .weather-name {
                font-size: .46rem;
            }
        }

        /* ================= V10 layout fixes ================= */

        /*
        The left rail is a core part of the product design.
        Keep it persistently available on desktop instead of allowing
        Streamlit to collapse it and strand the Reports navigation.
        */
        [data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            transform: none !important;
            min-width: 280px !important;
            width: 280px !important;
        }

        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] {
            display: none !important;
        }

        /* Make room for the persistent sidebar and prevent header clipping. */
        [data-testid="stMainBlockContainer"] {
            padding-top: 4rem !important;
        }

        /* Robust native rideability cards. */
        .ride-native-card {
            min-height: 132px;
            padding: .68rem .14rem .55rem .14rem;
            text-align: center;
            border-radius: 11px;
            background: rgba(7,14,18,.56);
            border: 1px solid rgba(255,255,255,.065);
            overflow: hidden;
        }

        .ride-native-card .ride-day-name {
            font-size: .60rem;
            color: var(--soft);
            font-weight: 780;
            text-transform: uppercase;
        }

        .ride-native-card .ride-score {
            font-size: 1.35rem;
            font-weight: 870;
            line-height: 1;
            margin: .55rem 0 .45rem 0;
        }

        .ride-rating-native {
            display: inline-flex !important;
            max-width: 100%;
            font-size: clamp(.46rem, .58vw, .60rem) !important;
            letter-spacing: 0 !important;
            padding: .20rem .24rem !important;
            white-space: nowrap !important;
        }

        .ride-surface {
            color: var(--faint);
            font-size: .53rem;
            font-weight: 720;
            margin-top: .38rem;
            text-transform: uppercase;
        }

        @media (max-width: 1050px) {
            .ride-native-card {
                min-height: 122px;
                padding-left: .06rem;
                padding-right: .06rem;
            }

            .ride-native-card .ride-score {
                font-size: 1.15rem;
            }

            .ride-rating-native {
                font-size: .44rem !important;
            }
        }

        /*
        On narrow/mobile screens let Streamlit handle the sidebar normally
        so it does not consume the whole viewport.
        */
        @media (max-width: 800px) {
            [data-testid="stSidebar"] {
                min-width: initial !important;
                width: initial !important;
            }

            [data-testid="stSidebarCollapseButton"],
            [data-testid="stSidebarCollapsedControl"] {
                display: flex !important;
            }
        }

        /* ================= V11 layout refinement ================= */

        .top-score-card {
            min-height: 330px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .graph-title {
            margin-top: .2rem !important;
            margin-bottom: .45rem !important;
        }

        .stats-card {
            margin-top: .95rem;
            margin-bottom: .7rem;
        }

        .lower-section-title {
            margin-top: .15rem !important;
        }

        .ranking-title {
            margin-top: 1.7rem !important;
        }

        .ranking-help {
            color: var(--faint);
            font-size: .73rem;
            padding-top: 2rem;
            text-align: right;
        }

        /* More breathing room in weather when it is below the hero row. */
        .weather-native-card {
            min-height: 144px !important;
            padding-left: .20rem !important;
            padding-right: .20rem !important;
        }

        .weather-native-card .weather-day-name {
            white-space: nowrap !important;
            font-size: .59rem !important;
        }

        .weather-native-card .weather-high {
            font-size: 1rem !important;
        }

        .weather-native-card .weather-low {
            font-size: .70rem !important;
        }

        .weather-native-card .weather-name {
            font-size: .49rem !important;
            line-height: 1.15 !important;
        }

        /* Rankings should read as a compact list, not giant spaced rows. */
        .rank-button {
            margin-bottom: .20rem !important;
        }

        .rank-button div.stButton > button {
            min-height: 42px !important;
            padding: .48rem .68rem !important;
            margin: 0 !important;
        }

        /* Do not render the old rideability-card module on Explore.
           The line graph is now the forecast visualization. */
        .ride-native-card {
            display: none !important;
        }

        @media (max-width: 1100px) {
            .top-score-card {
                min-height: 300px;
            }

            .score-grid {
                grid-template-columns: 116px 1fr !important;
            }

            .score-ring {
                width: 110px !important;
                height: 110px !important;
                border-width: 8px !important;
            }

            .score-number {
                font-size: 2.55rem !important;
            }
        }

        /* ================= V12 requested layout refinements ================= */

        .top-score-card {
            min-height: 0 !important;
            padding-bottom: 1rem !important;
        }

        .compact-trail-stats {
            display: grid;
            grid-template-columns: 1fr 1.35fr;
            gap: .55rem;
            border-top: 1px solid rgba(255,255,255,.08);
            margin-top: .85rem;
            padding-top: .75rem;
        }

        .compact-stat {
            min-width: 0;
        }

        .compact-stat-label {
            color: var(--faint);
            font-size: .60rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: .045em;
        }

        .compact-stat-value {
            font-size: .86rem;
            font-weight: 820;
            margin-top: .16rem;
            white-space: nowrap;
        }

        .compact-sun-card {
            margin-top: .55rem !important;
        }

        .graph-title {
            margin-top: .05rem !important;
            margin-bottom: .45rem !important;
        }

        .weather-section-title {
            margin-top: 1.35rem !important;
            margin-bottom: .45rem !important;
        }

        /* Full-width weather section: enough room for seven real columns. */
        .weather-native-card {
            min-height: 132px !important;
            padding: .62rem .24rem .52rem .24rem !important;
        }

        .weather-native-card .weather-day-name {
            white-space: nowrap !important;
            font-size: .63rem !important;
        }

        .weather-native-card .weather-icon {
            font-size: 1.65rem !important;
        }

        .weather-native-card .weather-high {
            font-size: 1.04rem !important;
        }

        .weather-native-card .weather-low {
            font-size: .72rem !important;
        }

        .weather-native-card .weather-rain {
            font-size: .57rem !important;
            white-space: nowrap !important;
        }

        .weather-native-card .weather-name {
            font-size: .53rem !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }

        /* Bring rankings back to the compact early-version feel. */
        .ranking-title {
            margin-top: 1.45rem !important;
            margin-bottom: .45rem !important;
        }

        div[data-testid="stButton"] {
            margin-bottom: .18rem !important;
        }

        div[data-testid="stButton"] > button {
            min-height: 42px !important;
            padding: .48rem .70rem !important;
            border-radius: 10px !important;
            margin: 0 !important;
            font-size: .79rem !important;
            font-weight: 720 !important;
        }

        /* Keep ranking buttons visually compact even in wide layout. */
        div[data-testid="stButton"] > button p {
            margin: 0 !important;
        }

        @media (max-width: 1100px) {
            .compact-trail-stats {
                grid-template-columns: 1fr;
            }

            .compact-stat-value {
                white-space: normal;
            }
        }

        /* ================= V13 hero balance refinements ================= */

        .matched-hero-card {
            min-height: 420px !important;
            box-sizing: border-box;
        }

        .score-card.matched-hero-card {
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding-top: 1.25rem !important;
            padding-bottom: 1.15rem !important;
        }

        .graph-card {
            padding: 1rem 1rem .9rem 1rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .graph-card-title {
            font-size: 1.12rem;
            font-weight: 830;
            letter-spacing: -.02em;
            margin-bottom: .1rem;
        }

        .graph-card-sub {
            color: var(--faint);
            font-size: .70rem;
            margin-bottom: .35rem;
        }

        /* Pull the pyplot visually into the graph card. */
        .graph-card + div[data-testid="stImage"],
        .graph-card + div[data-testid="stElementContainer"] {
            margin-top: 0 !important;
        }

        [data-testid="stImage"] {
            margin-top: .15rem !important;
            margin-bottom: .45rem !important;
            background: transparent !important;
            border: 0 !important;
            padding: 0 !important;
        }

        [data-testid="stImage"] img {
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,.07) !important;
        }

        .graph-summary-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: .5rem;
            border-top: 1px solid rgba(255,255,255,.08);
            padding-top: .7rem;
            margin-top: .15rem;
        }

        .graph-summary-item {
            min-width: 0;
        }

        .graph-summary-label {
            color: var(--faint);
            font-size: .58rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: .05em;
        }

        .graph-summary-value {
            font-size: .84rem;
            font-weight: 820;
            margin-top: .14rem;
            white-space: nowrap;
        }

        .full-width-sun-card {
            margin-top: .75rem !important;
            margin-bottom: .35rem !important;
            padding: .78rem .95rem !important;
        }

        @media (max-width: 1100px) {
            .matched-hero-card {
                min-height: 390px !important;
            }

            .graph-summary-strip {
                gap: .3rem;
            }

            .graph-summary-value {
                font-size: .76rem;
            }
        }

        /* ================= V14 native graph container ================= */

        /*
        The graph now lives in a real Streamlit bordered container.
        Style that container to visually match the score card.
        */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 18px !important;
            box-shadow: 0 18px 50px rgba(0,0,0,.24) !important;
            backdrop-filter: blur(15px);
            -webkit-backdrop-filter: blur(15px);
        }

        [data-testid="stVerticalBlockBorderWrapper"] > div {
            padding: 1rem 1rem .9rem 1rem !important;
        }

        /*
        Specifically match the hero graph container height to the score card
        without creating an empty HTML shell above the actual chart.
        */
        [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title) {
            min-height: 420px !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title) > div {
            min-height: 420px !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: center !important;
        }

        .graph-card-title {
            font-size: 1.12rem;
            font-weight: 830;
            letter-spacing: -.02em;
            margin: 0 0 .08rem 0;
        }

        .graph-card-sub {
            color: var(--faint);
            font-size: .70rem;
            margin: 0 0 .35rem 0;
        }

        .graph-summary-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: .5rem;
            border-top: 1px solid rgba(255,255,255,.08);
            padding-top: .7rem;
            margin-top: .05rem;
        }

        .graph-summary-item {
            min-width: 0;
        }

        .graph-summary-label {
            color: var(--faint);
            font-size: .58rem;
            font-weight: 760;
            text-transform: uppercase;
            letter-spacing: .05em;
        }

        .graph-summary-value {
            font-size: .84rem;
            font-weight: 820;
            margin-top: .14rem;
            white-space: nowrap;
        }

        @media (max-width: 1100px) {
            [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title),
            [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title) > div {
                min-height: 390px !important;
            }
        }

        /* ================= V15 graph rendering fix ================= */

        [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title)
        [data-testid="stImage"] {
            margin-top: .30rem !important;
            margin-bottom: .35rem !important;
            padding: 0 !important;
            background: transparent !important;
            border: 0 !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title)
        [data-testid="stImage"] img {
            width: 100% !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,.07) !important;
            display: block !important;
        }

        /* ================= V16 desktop-scale cleanup ================= */

        .matched-hero-card {
            min-height: 365px !important;
        }

        .score-card.matched-hero-card {
            padding: 1rem 1.05rem !important;
        }

        .score-grid {
            grid-template-columns: 128px 1fr !important;
            gap: .85rem !important;
        }

        .score-ring {
            width: 122px !important;
            height: 122px !important;
            border-width: 8px !important;
        }

        .score-number {
            font-size: 2.8rem !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title),
        [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title) > div {
            min-height: 365px !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title) > div {
            padding: .82rem .9rem .72rem .9rem !important;
        }

        .graph-card-title {
            font-size: 1.02rem !important;
        }

        .graph-card-sub {
            font-size: .66rem !important;
            margin-bottom: .16rem !important;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title)
        [data-testid="stImage"] {
            margin-top: .08rem !important;
            margin-bottom: .20rem !important;
        }

        .graph-summary-strip {
            padding-top: .52rem !important;
            margin-top: 0 !important;
        }

        .graph-summary-label {
            font-size: .54rem !important;
        }

        .graph-summary-value {
            font-size: .77rem !important;
        }

        .score-grid > div:last-child {
            align-self: center;
        }

        @media (min-width: 1350px) {
            [data-testid="stMainBlockContainer"] {
                max-width: 1280px !important;
            }
        }

        @media (max-width: 1100px) {
            .matched-hero-card,
            [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title),
            [data-testid="stVerticalBlockBorderWrapper"]:has(.graph-card-title) > div {
                min-height: 340px !important;
            }

            .score-grid {
                grid-template-columns: 112px 1fr !important;
            }

            .score-ring {
                width: 106px !important;
                height: 106px !important;
            }
        }

        /* ================= V17 final hero spacing ================= */

        .graph-sun-card {
            margin-top: .62rem !important;
            padding: .72rem .90rem !important;
            border-radius: 15px !important;
        }

        .graph-sun-card .sun-title {
            font-size: .82rem !important;
            margin-bottom: .16rem !important;
        }

        .graph-sun-card .sun-copy {
            font-size: .70rem !important;
            line-height: 1.35 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_prediction_data():
    prediction_path, _, _ = get_data_paths()

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{prediction_path}"
        )

    predictions = pd.read_csv(
        prediction_path,
        parse_dates=["date"],
    )

    return predictions


@st.cache_data(show_spinner=False)
def load_raw_forecast():
    _, raw_forecast_path, _ = get_data_paths()

    if not raw_forecast_path.exists():
        raise FileNotFoundError(
            f"Raw forecast file not found:\n{raw_forecast_path}"
        )

    raw = pd.read_csv(
        raw_forecast_path,
        parse_dates=["date"],
    )

    required = {
        "date",
        "trail_name",
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    }

    missing = required - set(raw.columns)

    if missing:
        raise RuntimeError(
            "Raw forecast file is missing: "
            + ", ".join(sorted(missing))
        )

    return raw


@st.cache_data(show_spinner=False)
def load_topography():
    _, _, topography_path = get_data_paths()

    if not topography_path.exists():
        return pd.DataFrame()

    return pd.read_csv(topography_path)


def merge_display_weather(predictions, raw):
    weather_columns = [
        "date",
        "trail_name",
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    ]

    display = predictions.merge(
        raw[weather_columns],
        on=[
            "date",
            "trail_name",
        ],
        how="left",
        suffixes=(
            "",
            "_raw",
        ),
        validate="one_to_one",
    )

    for base in [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
    ]:
        raw_column = f"{base}_raw"

        if raw_column in display.columns:
            display[base] = display[
                raw_column
            ].combine_first(
                display.get(base)
            )

            display = display.drop(
                columns=[raw_column]
            )

    return display


# ============================================================
# GOOGLE SHEETS
# ============================================================

def get_worksheet():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = Credentials.from_service_account_info(
        dict(st.secrets["google_service_account"]),
        scopes=scopes,
    )

    client = gspread.authorize(
        credentials
    )

    spreadsheet = client.open_by_key(
        st.secrets[
            "google_sheet"
        ][
            "spreadsheet_id"
        ]
    )

    return spreadsheet.worksheet(
        st.secrets[
            "google_sheet"
        ][
            "worksheet_name"
        ]
    )


def load_reports():
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

    try:
        records = (
            get_worksheet()
            .get_all_records()
        )

        if not records:
            return pd.DataFrame(
                columns=columns
            )

        reports = pd.DataFrame(
            records
        )

        for column in columns:
            if column not in reports.columns:
                reports[column] = ""

        return reports[
            columns
        ]

    except Exception:
        return pd.DataFrame(
            columns=columns
        )


def save_report(
    trail_name,
    condition,
    source,
    trail_section,
    notes,
):
    now = datetime.now(
        PARK_CITY_TIMEZONE
    )

    get_worksheet().append_row(
        [
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
        ],
        value_input_option="RAW",
    )


# ============================================================
# HELPERS
# ============================================================

def safe_number(
    value,
    digits=1,
):
    try:
        if pd.isna(value):
            return "—"

        return f"{float(value):.{digits}f}"

    except Exception:
        return "—"


def class_name(value):
    return (
        str(value)
        .lower()
        .replace(" ", "-")
    )


def sun_profile(row):
    north = float(
        row.get(
            "north_facing_pct",
            0,
        )
        or 0
    )

    east = float(
        row.get(
            "east_facing_pct",
            0,
        )
        or 0
    )

    south = float(
        row.get(
            "south_facing_pct",
            0,
        )
        or 0
    )

    west = float(
        row.get(
            "west_facing_pct",
            0,
        )
        or 0
    )

    if north >= 45:
        return (
            "🌲 Shadier · slower drying",
            "Mostly north-facing terrain receives less direct sun and tends to hold moisture longer."
        )

    if west >= 40 and south >= 25:
        return (
            "☀️ Strong afternoon sun · faster drying",
            "South- and west-facing terrain gets stronger afternoon sun and typically dries faster later in the day."
        )

    if east >= 40 and south >= 20:
        return (
            "🌤️ Early sunlight · faster morning drying",
            "East- and south-facing terrain warms earlier, helping morning moisture clear sooner."
        )

    if west >= 40:
        return (
            "☀️ Afternoon sunlight · later drying",
            "West-facing terrain gets its strongest direct sun later in the day."
        )

    if east >= 40:
        return (
            "🌤️ Morning sunlight · earlier drying",
            "East-facing terrain gets more early-day sun, helping damp sections begin drying earlier."
        )

    if south >= 40:
        return (
            "☀️ High sun exposure · faster drying",
            "Predominantly south-facing terrain receives strong direct sun and generally dries relatively quickly."
        )

    return (
        "🌤️ Mixed sun exposure · moderate drying",
        "Sunlight and drying vary by section throughout the day."
    )


def set_selected_trail(
    trail_name,
    zone_name,
):
    st.session_state[
        "selected_zone"
    ] = zone_name

    st.session_state[
        "selected_trail"
    ] = trail_name


def render_static_score_chart(
    trail_data,
):
    plot_data = (
        trail_data
        .sort_values("date")
        .copy()
    )

    labels = [
        pd.Timestamp(value).strftime("%b %d")
        for value in plot_data["date"]
    ]

    scores = (
        plot_data["rideability_score"]
        .astype(float)
        .tolist()
    )

    x_values = list(
        range(
            len(scores)
        )
    )

    y_min = max(
        0,
        min(scores) - 8,
    )

    background = "#0d1519"
    text_color = "#dce7e1"
    grid_color = "#344248"
    line_color = "#67cf73"

    fig, ax = plt.subplots(
        figsize=(9.4, 2.75),
        facecolor=background,
    )

    ax.set_facecolor(
        background
    )

    ax.fill_between(
        x_values,
        scores,
        y_min,
        color=line_color,
        alpha=.10,
    )

    ax.plot(
        x_values,
        scores,
        color=line_color,
        linewidth=2.6,
        marker="o",
        markersize=6.5,
        markerfacecolor=line_color,
        markeredgecolor="#d8f7dd",
        markeredgewidth=1.1,
    )

    # Score labels above each point.
    for x_value, score in zip(
        x_values,
        scores,
    ):
        ax.text(
            x_value,
            score + 1.4,
            f"{int(round(score))}",
            ha="center",
            va="bottom",
            color="#f4f7f5",
            fontsize=8.8,
            fontweight="bold",
        )

    ax.set_xticks(
        x_values
    )

    ax.set_xticklabels(
        labels,
        color=text_color,
        fontsize=8.5,
    )

    ax.set_ylim(
        y_min,
        100,
    )

    ax.set_ylabel(
        "Score",
        color=text_color,
        fontsize=9,
    )

    ax.tick_params(
        axis="y",
        colors=text_color,
        labelsize=8,
    )

    ax.tick_params(
        axis="x",
        colors=text_color,
        length=0,
    )

    ax.grid(
        axis="y",
        color=grid_color,
        alpha=.42,
        linewidth=.8,
    )

    ax.grid(
        axis="x",
        visible=False,
    )

    for spine in ax.spines.values():
        spine.set_color(
            grid_color
        )

    ax.spines["top"].set_visible(
        False
    )

    ax.spines["right"].set_visible(
        False
    )

    ax.margins(
        x=.04
    )

    fig.tight_layout(
        pad=.45
    )

    st.pyplot(
        fig,
        use_container_width=True,
        clear_figure=True,
    )

    plt.close(
        fig
    )


def render_weather_card(
    trail_data,
):
    ordered = (
        trail_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    current = ordered.iloc[0]

    with st.container(
        border=True,
    ):
        weather_columns = st.columns(
            len(ordered),
            gap="small",
        )

        for column, (_, row) in zip(
            weather_columns,
            ordered.iterrows(),
        ):
            icon, label = weather_code_info(
                row.get("weather_code"),
                row.get("precipitation_sum"),
            )

            date = pd.Timestamp(
                row["date"]
            )

            high = safe_number(
                row.get("temperature_2m_max"),
                0,
            )

            low = safe_number(
                row.get("temperature_2m_min"),
                0,
            )

            rain = safe_number(
                row.get("precipitation_sum"),
                2,
            )

            with column:
                st.markdown(
                    (
                        '<div class="weather-native-card">'
                        f'<div class="weather-day-name">{date.strftime("%a")}</div>'
                        f'<div class="weather-icon">{icon}</div>'
                        f'<div class="weather-high">{high}°</div>'
                        f'<div class="weather-low">{low}°</div>'
                        f'<div class="weather-rain">💧 {rain} in</div>'
                        f'<div class="weather-name">{label}</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

        stat_cols = st.columns(
            4,
            gap="small",
        )

        stat_values = [
            (
                "Today rain",
                f'{safe_number(current.get("precip_1d"), 2)} in',
            ),
            (
                "3-day rain",
                f'{safe_number(current.get("precip_3d"), 2)} in',
            ),
            (
                "7-day rain",
                f'{safe_number(current.get("precip_7d"), 2)} in',
            ),
            (
                "Dry days",
                safe_number(
                    current.get("days_since_precip"),
                    0,
                ),
            ),
        ]

        for column, (
            label,
            value,
        ) in zip(
            stat_cols,
            stat_values,
        ):
            with column:
                st.markdown(
                    (
                        '<div class="weather-stat-native">'
                        f'<div class="weather-stat-label">{label}</div>'
                        f'<div class="weather-stat-value">{value}</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )


def render_rideability_strip(
    trail_data,
):
    ordered = (
        trail_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    with st.container(
        border=True,
    ):
        st.markdown(
            '<div class="section-title" style="margin-top:0;">7-day rideability forecast</div>',
            unsafe_allow_html=True,
        )

        ride_columns = st.columns(
            len(ordered),
            gap="small",
        )

        for column, (_, row) in zip(
            ride_columns,
            ordered.iterrows(),
        ):
            date = pd.Timestamp(
                row["date"]
            )

            score = int(
                round(
                    row["rideability_score"]
                )
            )

            rating = str(
                row["rideability"]
            )

            surface = str(
                row["surface_state"]
            )

            with column:
                st.markdown(
                    (
                        '<div class="ride-native-card">'
                        f'<div class="ride-day-name">{date.strftime("%a")}</div>'
                        f'<div class="ride-score">{score}</div>'
                        f'<div class="tag {class_name(rating)} ride-rating-native">{rating}</div>'
                        f'<div class="ride-surface">{surface}</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )


def render_recent_reports_preview(
    reports,
    selected_trail,
):
    if reports.empty:
        st.markdown(
            (
                '<div class="glass-card report-preview-card">'
                '<div class="report-preview-title">'
                'No rider reports yet'
                '</div>'
                '<div class="report-preview-copy">'
                'Submit the first ground-truth report for this trail.'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        return

    trail_reports = (
        reports[
            reports[
                "trail_name"
            ].astype(str)
            == selected_trail
        ]
        .tail(3)
        .iloc[::-1]
    )

    if trail_reports.empty:
        st.markdown(
            (
                '<div class="glass-card report-preview-card">'
                '<div class="report-preview-title">'
                'No recent reports for this trail'
                '</div>'
                '<div class="report-preview-copy">'
                'Use Reports in the sidebar to add one.'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

        return

    for _, report in (
        trail_reports
        .iterrows()
    ):
        section = str(
            report.get(
                "trail_section",
                "",
            )
        ).strip()

        title = (
            selected_trail
            + (
                f" · {section}"
                if section
                else ""
            )
        )

        condition = str(
            report.get(
                "condition",
                "",
            )
        ).upper()

        notes = str(
            report.get(
                "notes",
                "",
            )
        ).strip()

        meta = (
            f"{report.get('date','')} · "
            f"{report.get('time','')} · "
            f"{str(report.get('source','')).title()}"
        )

        st.markdown(
            (
                '<div class="glass-card report-preview-card">'
                f'<div class="report-preview-title">{title}</div>'
                f'<div style="margin-top:.28rem;"><span class="tag {class_name(condition)}">{condition}</span></div>'
                f'<div class="report-preview-meta">{meta}</div>'
                f'<div class="report-preview-copy">{notes if notes else "No notes provided."}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


def render_rankings(
    today_data,
    zone_filter,
    limit=8,
):
    ranking = today_data.copy()

    if zone_filter != "All zones":
        ranking = ranking[
            ranking[
                "final_area"
            ]
            == zone_filter
        ]

    ranking = (
        ranking
        .sort_values(
            [
                "rideability_score",
                "trail_name",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .head(limit)
        .reset_index(
            drop=True
        )
    )

    for idx, row in ranking.iterrows():
        label = (
            f"{idx + 1}. "
            f"{row['trail_name']}   "
            f"· {int(round(row['rideability_score']))}/100 "
            f"· {row['rideability']}"
        )

        if st.button(
            label,
            key=(
                "rank_"
                + str(
                    row[
                        "trail_name"
                    ]
                )
            ),
            use_container_width=True,
            on_click=set_selected_trail,
            args=(
                row[
                    "trail_name"
                ],
                row[
                    "final_area"
                ],
            ),
        ):
            pass


# ============================================================
# REPORTS PAGE
# ============================================================

def render_reports_page(
    forecast,
    selected_trail,
    current,
):
    st.markdown(
        '<div class="zone-kicker">Community ground truth</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="report-page-title">Rider Reports</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="trail-date">Compare the model forecast with what riders are actually finding on trail.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="glass-card report-card">'
            f'<div class="report-title">Model forecast · {selected_trail}</div>'
            '<div style="display:flex;gap:.45rem;align-items:center;flex-wrap:wrap;margin-top:.55rem;">'
            f'<span class="tag {class_name(current["surface_state"])}">{current["surface_state"]}</span>'
            f'<span class="tag {class_name(current["rideability"])}">{current["rideability"]}</span>'
            f'<strong>{int(round(current["rideability_score"]))}/100</strong>'
            '</div>'
            '<div class="report-copy">What did you actually find?</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    condition = st.radio(
        "Surface condition",
        ALLOWED_CONDITIONS,
        horizontal=True,
    )

    source = st.radio(
        "Report source",
        REPORT_SOURCES,
        horizontal=True,
    )

    trail_section = st.text_input(
        "Trail section (optional)",
        placeholder=(
            "Upper, lower, shaded section, etc."
        ),
    )

    notes = st.text_area(
        "Notes (optional)",
        placeholder=(
            "Example: tacky overall, "
            "one damp corner near the trees"
        ),
    )

    if st.button(
        "Submit report",
        type="primary",
        use_container_width=True,
    ):
        try:
            save_report(
                trail_name=selected_trail,
                condition=condition,
                source=source,
                trail_section=trail_section,
                notes=notes,
            )

            st.success(
                f"Saved: "
                f"{selected_trail} — "
                f"{condition}"
            )

        except Exception as error:
            st.error(
                "The report could not be saved."
            )
            st.exception(
                error
            )

    st.markdown(
        '<div class="section-title">Recent reports</div>',
        unsafe_allow_html=True,
    )

    reports = load_reports()

    if reports.empty:
        st.caption(
            "No rider reports yet."
        )
        return

    recent = (
        reports
        .tail(15)
        .iloc[::-1]
        .copy()
    )

    for _, report in (
        recent
        .iterrows()
    ):
        trail_name = str(
            report.get(
                "trail_name",
                "",
            )
        )

        section = str(
            report.get(
                "trail_section",
                "",
            )
        ).strip()

        title = (
            trail_name
            + (
                f" · {section}"
                if section
                else ""
            )
        )

        condition_text = str(
            report.get(
                "condition",
                "",
            )
        ).upper()

        notes_text = str(
            report.get(
                "notes",
                "",
            )
        ).strip()

        st.markdown(
            (
                '<div class="glass-card report-card">'
                f'<div class="report-title">{title}</div>'
                f'<div style="margin-top:.35rem;"><span class="tag {class_name(condition_text)}">{condition_text}</span></div>'
                f'<div class="report-meta">{report.get("date","")} · {report.get("time","")} · {str(report.get("source","")).title()}</div>'
                f'<div class="report-copy">{notes_text if notes_text else "No notes provided."}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


# ============================================================
# MAIN
# ============================================================

def main():
    st.set_page_config(
        page_title=(
            "Park City Trail Conditions"
        ),
        page_icon="🚵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_css()

    try:
        predictions = (
            load_prediction_data()
        )

        raw_forecast = (
            load_raw_forecast()
        )

        forecast = (
            merge_display_weather(
                predictions,
                raw_forecast,
            )
        )

    except Exception as error:
        st.error(
            "Forecast data could not be loaded."
        )
        st.exception(
            error
        )
        st.stop()

    topography = (
        load_topography()
    )

    forecast_dates = sorted(
        forecast[
            "date"
        ]
        .dropna()
        .unique()
    )

    first_date = pd.Timestamp(
        forecast_dates[0]
    )

    today_data = forecast[
        forecast[
            "date"
        ]
        == first_date
    ].copy()

    zones = sorted(
        forecast[
            "final_area"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    all_trails = sorted(
        forecast[
            "trail_name"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if (
        "active_page"
        not in st.session_state
    ):
        st.session_state[
            "active_page"
        ] = "Explore"

    if (
        "selected_zone"
        not in st.session_state
    ):
        st.session_state[
            "selected_zone"
        ] = "All zones"

    if (
        "selected_trail"
        not in st.session_state
    ):
        st.session_state[
            "selected_trail"
        ] = all_trails[0]

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    with st.sidebar:
        st.markdown(
            """
            <div class="brand">
                <div class="brand-main">
                    <span class="brand-mark">△</span>
                    PARK CITY
                </div>
                <div class="brand-sub">
                    Trail Conditions
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="nav-label">Navigate</div>',
            unsafe_allow_html=True,
        )

        st.radio(
            "Navigation",
            [
                "Explore",
                "Reports",
            ],
            key="active_page",
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="nav-label">Select location</div>',
            unsafe_allow_html=True,
        )

        zone_options = (
            ["All zones"]
            + zones
        )

        if (
            st.session_state[
                "selected_zone"
            ]
            not in zone_options
        ):
            st.session_state[
                "selected_zone"
            ] = "All zones"

        zone_choice = st.selectbox(
            "Zone",
            zone_options,
            key="selected_zone",
        )

        if (
            zone_choice
            == "All zones"
        ):
            trail_options = (
                all_trails
            )

        else:
            trail_options = sorted(
                forecast[
                    forecast[
                        "final_area"
                    ]
                    == zone_choice
                ][
                    "trail_name"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        if (
            st.session_state[
                "selected_trail"
            ]
            not in trail_options
        ):
            st.session_state[
                "selected_trail"
            ] = trail_options[0]

        selected_trail = (
            st.selectbox(
                "Trail",
                trail_options,
                key="selected_trail",
            )
        )

        st.divider()

        st.caption(
            "Weather + terrain + rider reports"
        )

    # --------------------------------------------------------
    # SELECTED TRAIL DATA
    # --------------------------------------------------------

    trail_data = (
        forecast[
            forecast[
                "trail_name"
            ]
            == selected_trail
        ]
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )

    current = (
        trail_data
        .iloc[0]
    )

    if (
        st.session_state[
            "active_page"
        ]
        == "Reports"
    ):
        render_reports_page(
            forecast,
            selected_trail,
            current,
        )
        st.stop()

    reports = load_reports()

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        (
            '<div class="zone-kicker">'
            f'{current["final_area"]} zone'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="trail-title">'
            f'{selected_trail}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        (
            '<div class="trail-date">'
            'Forecast beginning '
            f'{pd.Timestamp(current["date"]).strftime("%b %d, %Y")}'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # HERO ROW — MATCHED SCORE + GRAPH CARDS
    # --------------------------------------------------------

    score = int(
        round(
            current[
                "rideability_score"
            ]
        )
    )

    rating = str(
        current[
            "rideability"
        ]
    )

    surface = str(
        current[
            "surface_state"
        ]
    )

    surface_description = (
        SURFACE_HELP.get(
            surface.upper(),
            "Modeled trail-condition estimate",
        )
    )

    terrain_row = pd.Series(
        dtype="object"
    )

    if not topography.empty:
        match = topography[
            topography[
                "trail_name"
            ]
            == selected_trail
        ]

        if not match.empty:
            terrain_row = (
                match.iloc[0]
            )

    distance = "—"
    elevation_range = "—"
    sun_title = None
    sun_copy = None

    if not terrain_row.empty:
        distance = (
            f'{safe_number(terrain_row.get("sampled_length_miles"), 1)} mi'
        )

        minimum = safe_number(
            terrain_row.get(
                "minimum_elevation_feet"
            ),
            0,
        )

        maximum = safe_number(
            terrain_row.get(
                "maximum_elevation_feet"
            ),
            0,
        )

        elevation_range = (
            f"{minimum}–{maximum} ft"
        )

        sun_title, sun_copy = (
            sun_profile(
                terrain_row
            )
        )

    ordered_scores = (
        trail_data
        .sort_values("date")
        .reset_index(drop=True)
    )

    best_idx = (
        ordered_scores[
            "rideability_score"
        ]
        .astype(float)
        .idxmax()
    )

    best_row = ordered_scores.loc[
        best_idx
    ]

    best_day = pd.Timestamp(
        best_row["date"]
    ).strftime("%a")

    best_score = int(
        round(
            best_row[
                "rideability_score"
            ]
        )
    )

    weekend_rows = ordered_scores[
        pd.to_datetime(
            ordered_scores["date"]
        ).dt.dayofweek.isin(
            [5, 6]
        )
    ]

    if weekend_rows.empty:
        weekend_text = "—"
    else:
        weekend_text = " / ".join(
            str(
                int(
                    round(
                        value
                    )
                )
            )
            for value in weekend_rows[
                "rideability_score"
            ].tolist()
        )

    hero_left, hero_right = st.columns(
        [
            .86,
            1.14,
        ],
        gap="large",
    )

    with hero_left:
        st.markdown(
            (
                '<div class="glass-card score-card matched-hero-card">'
                '<div class="score-grid">'
                '<div class="score-ring">'
                f'<div class="score-number">{score}</div>'
                '<div class="score-denom">/100</div>'
                '</div>'
                '<div>'
                f'<span class="tag {class_name(surface)}">{surface}</span>'
                '<div style="height:.38rem;"></div>'
                f'<span class="tag {class_name(rating)}">{rating}</span>'
                '</div>'
                '</div>'
                '<div class="score-summary">'
                f'<strong>{surface_description}</strong><br>'
                f'{current["reason"]}'
                '</div>'
                '<div class="compact-trail-stats">'
                '<div class="compact-stat">'
                '<div class="compact-stat-label">Distance</div>'
                f'<div class="compact-stat-value">{distance}</div>'
                '</div>'
                '<div class="compact-stat">'
                '<div class="compact-stat-label">Elevation range</div>'
                f'<div class="compact-stat-value">{elevation_range}</div>'
                '</div>'
                '</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    with hero_right:
        with st.container(
            border=True,
        ):
            st.markdown(
                '<div class="graph-card-title">7-day rideability trend</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                '<div class="graph-card-sub">Forecasted rideability score</div>',
                unsafe_allow_html=True,
            )

            render_static_score_chart(
                trail_data
            )

            st.markdown(
                (
                    '<div class="graph-summary-strip">'
                    '<div class="graph-summary-item">'
                    '<div class="graph-summary-label">Best day</div>'
                    f'<div class="graph-summary-value">{best_day} · {best_score}</div>'
                    '</div>'
                    '<div class="graph-summary-item">'
                    '<div class="graph-summary-label">Weekend</div>'
                    f'<div class="graph-summary-value">{weekend_text}</div>'
                    '</div>'
                    '<div class="graph-summary-item">'
                    '<div class="graph-summary-label">Scale</div>'
                    '<div class="graph-summary-value">0–100</div>'
                    '</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

        if (
            sun_title is not None
            and sun_copy is not None
        ):
            st.markdown(
                (
                    '<div class="glass-card sun-card graph-sun-card">'
                    f'<div class="sun-title">{sun_title}</div>'
                    f'<div class="sun-copy">{sun_copy}</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # 7-DAY WEATHER — FULL WIDTH
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title weather-section-title">7-day weather forecast</div>',
        unsafe_allow_html=True,
    )

    render_weather_card(
        trail_data
    )

    # --------------------------------------------------------
    # BEST RIDING — COMPACT FULL-WIDTH SECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title ranking-title">Best riding today</div>',
        unsafe_allow_html=True,
    )

    ranking_zone = st.selectbox(
        "Ranking zone",
        [
            "All zones"
        ]
        + zones,
        index=(
            0
            if zone_choice
            == "All zones"
            else (
                [
                    "All zones"
                ]
                + zones
            ).index(
                zone_choice
            )
        ),
        key="ranking_zone",
    )

    render_rankings(
        today_data,
        ranking_zone,
        limit=8,
    )


if __name__ == "__main__":
    main()