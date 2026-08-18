from pathlib import Path

import pandas as pd


ALLOWED_CONDITIONS = {
    "dry",
    "ideal",
    "wet",
    "muddy",
    "snow",
}

ALLOWED_SOURCES = {
    "customer",
    "staff",
    "personal",
}

REQUIRED_COLUMNS = {
    "date",
    "time",
    "trail_name",
    "condition",
    "source",
    "trail_section",
    "notes",
}


def main():
    project_root = Path(__file__).resolve().parents[2]

    reports_path = (
        project_root
        / "data"
        / "observations"
        / "trail_condition_reports.csv"
    )

    catalog_path = (
        project_root
        / "data"
        / "processed"
        / "clean_trail_catalog.csv"
    )

    print("Loading trail condition reports...")

    reports = pd.read_csv(
        reports_path,
        keep_default_na=False,
    )

    print(f"Report rows found: {len(reports):,}")

    print("Loading approved trail catalog...")

    catalog = pd.read_csv(catalog_path)

    approved_trails = set(
        catalog["trail_name"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    errors = []

    # ---------------------------------------------------------
    # REQUIRED COLUMNS
    # ---------------------------------------------------------

    missing_columns = (
        REQUIRED_COLUMNS
        - set(reports.columns)
    )

    if missing_columns:
        print("\nValidation failed:")
        print(
            "- Missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
        return

    # ---------------------------------------------------------
    # EMPTY DATASET
    # ---------------------------------------------------------

    if reports.empty:
        print("\nValidation passed!")
        print("Reports: 0")
        print(
            "The observation file is ready "
            "for real reports."
        )
        return

    # ---------------------------------------------------------
    # NORMALIZE TEXT
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # DATE
    # ---------------------------------------------------------

    parsed_dates = pd.to_datetime(
        reports["date"],
        format="%Y-%m-%d",
        errors="coerce",
    )

    invalid_date_count = (
        parsed_dates.isna().sum()
    )

    if invalid_date_count:
        errors.append(
            f"{invalid_date_count} row(s) "
            "have invalid dates. "
            "Use YYYY-MM-DD."
        )

    # ---------------------------------------------------------
    # TIME
    # ---------------------------------------------------------

    time_pattern = (
        r"^(?:[01]\d|2[0-3]):[0-5]\d$"
    )

    valid_times = (
        reports["time"]
        .astype(str)
        .str.match(
            time_pattern,
            na=False,
        )
    )

    invalid_time_count = (
        (~valid_times).sum()
    )

    if invalid_time_count:
        errors.append(
            f"{invalid_time_count} row(s) "
            "have invalid times. "
            "Use HH:MM, for example 14:30."
        )

    # ---------------------------------------------------------
    # TRAILS
    # ---------------------------------------------------------

    invalid_trails = reports[
        ~reports["trail_name"]
        .isin(approved_trails)
    ]

    if not invalid_trails.empty:
        bad_trails = sorted(
            invalid_trails[
                "trail_name"
            ].unique()
        )

        errors.append(
            "Unknown trail name(s): "
            + ", ".join(bad_trails)
        )

    # ---------------------------------------------------------
    # CONDITIONS
    # ---------------------------------------------------------

    invalid_conditions = reports[
        ~reports["condition"]
        .isin(ALLOWED_CONDITIONS)
    ]

    if not invalid_conditions.empty:
        bad_conditions = sorted(
            invalid_conditions[
                "condition"
            ].unique()
        )

        errors.append(
            "Invalid condition value(s): "
            + ", ".join(bad_conditions)
        )

    # ---------------------------------------------------------
    # SOURCES
    # ---------------------------------------------------------

    invalid_sources = reports[
        ~reports["source"]
        .isin(ALLOWED_SOURCES)
    ]

    if not invalid_sources.empty:
        bad_sources = sorted(
            invalid_sources[
                "source"
            ].unique()
        )

        errors.append(
            "Invalid source value(s): "
            + ", ".join(bad_sources)
        )

    # ---------------------------------------------------------
    # EXACT DUPLICATES
    #
    # Multiple reports for the same trail/date are GOOD.
    # We only flag identical entries.
    # ---------------------------------------------------------

    duplicate_count = reports.duplicated(
        subset=[
            "date",
            "time",
            "trail_name",
            "condition",
            "source",
            "trail_section",
            "notes",
        ]
    ).sum()

    if duplicate_count:
        errors.append(
            f"{duplicate_count} exact "
            "duplicate report row(s) found."
        )

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    if errors:
        print("\nValidation found issues:")

        for error in errors:
            print(f"- {error}")

        return

    print("\nValidation passed!")

    print(
        f"Reports: {len(reports):,}"
    )

    print(
        "Unique trails reported: "
        f"{reports['trail_name'].nunique()}"
    )

    print(
        "First report date: "
        f"{parsed_dates.min().date()}"
    )

    print(
        "Last report date: "
        f"{parsed_dates.max().date()}"
    )

    print("\nReports by condition:")

    print(
        reports["condition"]
        .value_counts()
        .to_string()
    )

    print("\nReports by source:")

    print(
        reports["source"]
        .value_counts()
        .to_string()
    )


if __name__ == "__main__":
    main()