from pathlib import Path

import pandas as pd


VALID_CONDITIONS = {
    "ideal",
    "dry",
    "wet",
    "muddy",
    "snow",
}


def main():
    project_root = Path(__file__).resolve().parents[2]

    reports_path = (
        project_root
        / "data"
        / "raw"
        / "condition_reports.csv"
    )

    modeling_path = (
        project_root
        / "data"
        / "processed"
        / "trail_modeling_dataset_v2.csv"
    )

    output_path = (
        project_root
        / "data"
        / "processed"
        / "labeled_trail_conditions.csv"
    )

    # ---------------------------------------------------------
    # LOAD REPORTS
    # ---------------------------------------------------------

    print("Loading condition reports...")

    reports = pd.read_csv(
        reports_path,
        keep_default_na=False,
    )

    print(f"Reports found: {len(reports):,}")

    if reports.empty:
        print(
            "\nNo condition reports exist yet."
        )
        print(
            "Nothing to join to the modeling dataset."
        )
        return

    # ---------------------------------------------------------
    # CLEAN REPORTS
    # ---------------------------------------------------------

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

    # Remove reports with unusable dates.
    invalid_dates = reports["date"].isna().sum()

    if invalid_dates:
        print(
            f"Removing {invalid_dates} report(s) "
            "with invalid dates."
        )

        reports = reports[
            reports["date"].notna()
        ].copy()

    # Remove unknown condition labels.
    invalid_conditions = (
        ~reports["condition"].isin(
            VALID_CONDITIONS
        )
    )

    invalid_condition_count = (
        invalid_conditions.sum()
    )

    if invalid_condition_count:
        print(
            f"Removing "
            f"{invalid_condition_count} report(s) "
            "with invalid conditions."
        )

        reports = reports[
            ~invalid_conditions
        ].copy()

    if reports.empty:
        print(
            "\nNo usable condition reports "
            "remain after validation."
        )
        return

    # ---------------------------------------------------------
    # LOAD MODELING DATA
    # ---------------------------------------------------------

    print("Loading modeling dataset...")

    modeling = pd.read_csv(
        modeling_path
    )

    print(
        f"Modeling rows: {len(modeling):,}"
    )

    modeling["date"] = pd.to_datetime(
        modeling["date"],
        errors="coerce",
    )

    modeling["trail_name"] = (
        modeling["trail_name"]
        .astype(str)
        .str.strip()
    )

    # ---------------------------------------------------------
    # JOIN REPORTS TO WEATHER + TERRAIN
    # ---------------------------------------------------------

    print(
        "Joining reports to weather "
        "and terrain features..."
    )

    labeled = reports.merge(
        modeling,
        on=[
            "trail_name",
            "date",
        ],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    matched = (
        labeled["_merge"] == "both"
    ).sum()

    unmatched = (
        labeled["_merge"] == "left_only"
    ).sum()

    print(
        f"Matched reports: {matched:,}"
    )

    print(
        f"Unmatched reports: {unmatched:,}"
    )

    # ---------------------------------------------------------
    # SHOW UNMATCHED REPORTS
    # ---------------------------------------------------------

    if unmatched:
        print(
            "\nReports without matching "
            "weather/terrain data:"
        )

        print(
            labeled.loc[
                labeled["_merge"]
                == "left_only",
                [
                    "date",
                    "trail_name",
                    "condition",
                ],
            ].to_string(
                index=False
            )
        )

    # For ML training, only keep observations
    # with corresponding feature data.
    labeled = labeled[
        labeled["_merge"] == "both"
    ].copy()

    labeled.drop(
        columns="_merge",
        inplace=True,
    )

    # ---------------------------------------------------------
    # RENAME LABEL
    # ---------------------------------------------------------

    labeled.rename(
        columns={
            "condition":
                "actual_condition"
        },
        inplace=True,
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    labeled.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print(
        "\nLabeled modeling dataset created!"
    )

    print(
        f"Rows: {len(labeled):,}"
    )

    print(
        f"Trails represented: "
        f"{labeled['trail_name'].nunique()}"
    )

    print(
        f"Columns: {len(labeled.columns)}"
    )

    print(
        f"Saved to: {output_path}"
    )

    if not labeled.empty:

        print(
            "\nCondition labels:"
        )

        print(
            labeled[
                "actual_condition"
            ]
            .value_counts()
            .to_string()
        )

        print(
            "\nReports by source:"
        )

        print(
            labeled[
                "source"
            ]
            .value_counts()
            .to_string()
        )

        preview_columns = [
            "date",
            "trail_name",
            "actual_condition",
            "source",
            "precip_1d",
            "precip_3d",
            "precip_7d",
            "days_since_precip",
            "temperature_2m_mean",
            "mean_slope_degrees",
            "north_facing_pct",
            "south_facing_pct",
        ]

        preview_columns = [
            column
            for column
            in preview_columns
            if column in labeled.columns
        ]

        print(
            "\nExample labeled observations:"
        )

        print(
            labeled[
                preview_columns
            ]
            .head(10)
            .to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
