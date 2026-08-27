from pathlib import Path
from datetime import date, timedelta
import time

import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

OPEN_METEO_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)

START_DATE = "2023-01-01"

HISTORICAL_LAG_DAYS = 5

# Smaller batches reduce the size/cost of each archive request.
BATCH_SIZE = 10

MAX_RETRIES = 10

REQUEST_TIMEOUT = 180

# Deliberately slow between successful requests.
SUCCESS_WAIT_SECONDS = 15

# Longer cooldown after rate limiting.
RATE_LIMIT_BASE_WAIT = 30

METERS_PER_FOOT = 0.3048


DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
]


OUTPUT_COLUMNS = [
    "date",
    "trail_name",
    "final_area",
    "trail_latitude",
    "trail_longitude",
    "trail_elevation_feet",
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
]


# ============================================================
# PATHS
# ============================================================


def get_project_paths():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    catalog_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_catalog.csv"
    )

    topography_path = (
        project_root
        / "data"
        / "processed"
        / "master_trail_topography_features.csv"
    )

    output_path = (
        project_root
        / "data"
        / "raw"
        / "master_trail_historical_weather.csv"
    )

    progress_path = (
        project_root
        / "data"
        / "raw"
        / "master_weather_download_progress.csv"
    )

    return (
        catalog_path,
        topography_path,
        output_path,
        progress_path,
    )


# ============================================================
# HELPERS
# ============================================================


def get_end_date():

    return (
        date.today()
        - timedelta(
            days=HISTORICAL_LAG_DAYS
        )
    ).isoformat()


def chunk_dataframe(
    dataframe,
    size,
):

    for start in range(
        0,
        len(dataframe),
        size,
    ):

        yield dataframe.iloc[
            start:start + size
        ].copy()


def load_existing_progress(
    progress_path,
):

    if not progress_path.exists():

        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    print(
        "Existing download progress found."
    )

    progress = pd.read_csv(
        progress_path,
        parse_dates=["date"],
    )

    print(
        f"Previously downloaded rows: "
        f"{len(progress):,}"
    )

    print(
        f"Previously completed trails: "
        f"{progress['trail_name'].nunique():,}"
    )

    return progress


def save_progress(
    dataframes,
    progress_path,
):

    if not dataframes:
        return

    combined = pd.concat(
        dataframes,
        ignore_index=True,
    )

    combined = (
        combined
        .drop_duplicates(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sort_values(
            [
                "trail_name",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    progress_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        progress_path,
        index=False,
    )


def request_batch(
    batch,
    start_date,
    end_date,
):

    latitudes = (
        batch["latitude"]
        .astype(float)
        .tolist()
    )

    longitudes = (
        batch["longitude"]
        .astype(float)
        .tolist()
    )

    elevation_meters = (
        batch["mean_elevation_feet"]
        .astype(float)
        * METERS_PER_FOOT
    ).tolist()

    params = {
        "latitude":
            ",".join(
                f"{value:.6f}"
                for value in latitudes
            ),

        "longitude":
            ",".join(
                f"{value:.6f}"
                for value in longitudes
            ),

        "elevation":
            ",".join(
                f"{value:.1f}"
                for value in elevation_meters
            ),

        "start_date":
            start_date,

        "end_date":
            end_date,

        "daily":
            ",".join(
                DAILY_VARIABLES
            ),

        "temperature_unit":
            "fahrenheit",

        "precipitation_unit":
            "inch",

        "timezone":
            "America/Denver",
    }

    for attempt in range(
        MAX_RETRIES
    ):

        try:

            response = requests.get(
                OPEN_METEO_URL,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:

                return response.json()

            if response.status_code == 429:

                wait_seconds = min(
                    300,
                    RATE_LIMIT_BASE_WAIT
                    * (
                        attempt + 1
                    ),
                )

                print(
                    "  Rate limited (429). "
                    f"Cooling down for "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            if response.status_code in {
                500,
                502,
                503,
                504,
            }:

                wait_seconds = min(
                    120,
                    10
                    * (
                        2 ** attempt
                    ),
                )

                print(
                    f"  Temporary API response "
                    f"{response.status_code}. "
                    f"Waiting "
                    f"{wait_seconds} seconds..."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            response.raise_for_status()

        except requests.RequestException as error:

            if (
                attempt
                == MAX_RETRIES - 1
            ):
                raise

            wait_seconds = min(
                120,
                10
                * (
                    2 ** attempt
                ),
            )

            print(
                f"  Request error: "
                f"{error}"
            )

            print(
                f"  Waiting "
                f"{wait_seconds} seconds..."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "Open-Meteo request failed "
        "after all retries."
    )


def normalize_response(
    response_data,
    expected_locations,
):

    if isinstance(
        response_data,
        dict,
    ):

        responses = [
            response_data
        ]

    elif isinstance(
        response_data,
        list,
    ):

        responses = (
            response_data
        )

    else:

        raise RuntimeError(
            "Unexpected Open-Meteo "
            "response format."
        )

    if (
        len(responses)
        != expected_locations
    ):

        raise RuntimeError(
            "Open-Meteo response location count "
            "does not match request count. "
            f"Expected "
            f"{expected_locations}, "
            f"received "
            f"{len(responses)}."
        )

    return responses


def parse_location_response(
    trail,
    response_data,
):

    if (
        "daily"
        not in response_data
    ):

        raise RuntimeError(
            f"No daily weather returned "
            f"for "
            f"{trail['trail_name']}."
        )

    daily = pd.DataFrame(
        response_data[
            "daily"
        ]
    )

    if daily.empty:

        raise RuntimeError(
            f"Empty weather response "
            f"for "
            f"{trail['trail_name']}."
        )

    daily = daily.rename(
        columns={
            "time":
                "date"
        }
    )

    daily["date"] = (
        pd.to_datetime(
            daily["date"]
        )
    )

    daily[
        "trail_name"
    ] = trail[
        "trail_name"
    ]

    daily[
        "final_area"
    ] = trail[
        "final_area"
    ]

    daily[
        "trail_latitude"
    ] = float(
        trail[
            "latitude"
        ]
    )

    daily[
        "trail_longitude"
    ] = float(
        trail[
            "longitude"
        ]
    )

    daily[
        "trail_elevation_feet"
    ] = float(
        trail[
            "mean_elevation_feet"
        ]
    )

    return daily[
        OUTPUT_COLUMNS
    ]


# ============================================================
# MAIN
# ============================================================


def main():

    (
        catalog_path,
        topography_path,
        output_path,
        progress_path,
    ) = get_project_paths()

    end_date = get_end_date()

    # ---------------------------------------------------------
    # LOAD CATALOG
    # ---------------------------------------------------------

    print(
        "Loading master trail catalog..."
    )

    catalog = pd.read_csv(
        catalog_path
    )

    print(
        f"Catalog trails: "
        f"{len(catalog):,}"
    )

    # ---------------------------------------------------------
    # LOAD TOPOGRAPHY
    # ---------------------------------------------------------

    print(
        "Loading master topography features..."
    )

    topography = pd.read_csv(
        topography_path
    )

    print(
        f"Topography trails: "
        f"{len(topography):,}"
    )

    # ---------------------------------------------------------
    # BUILD WEATHER LOCATION TABLE
    # ---------------------------------------------------------

    weather_locations = (
        catalog[
            [
                "trail_name",
                "final_area",
                "latitude",
                "longitude",
            ]
        ]
        .merge(
            topography[
                [
                    "trail_name",
                    "mean_elevation_feet",
                ]
            ],
            on="trail_name",
            how="left",
            validate="one_to_one",
        )
    )

    # ---------------------------------------------------------
    # VALIDATE
    # ---------------------------------------------------------

    duplicate_names = (
        weather_locations[
            "trail_name"
        ]
        .duplicated()
        .sum()
    )

    missing_coordinates = (
        weather_locations[
            [
                "latitude",
                "longitude",
            ]
        ]
        .isna()
        .any(
            axis=1
        )
        .sum()
    )

    missing_elevation = (
        weather_locations[
            "mean_elevation_feet"
        ]
        .isna()
        .sum()
    )

    print()
    print(
        f"Duplicate trail names: "
        f"{duplicate_names:,}"
    )

    print(
        f"Missing coordinates: "
        f"{missing_coordinates:,}"
    )

    print(
        f"Missing elevations: "
        f"{missing_elevation:,}"
    )

    if duplicate_names > 0:
        raise RuntimeError(
            "Duplicate trail names found."
        )

    if missing_coordinates > 0:
        raise RuntimeError(
            "Some trails are missing coordinates."
        )

    if missing_elevation > 0:
        raise RuntimeError(
            "Some trails are missing elevation."
        )

    # ---------------------------------------------------------
    # LOAD PREVIOUS PROGRESS
    # ---------------------------------------------------------

    progress = load_existing_progress(
        progress_path
    )

    completed_trails = set(
        progress[
            "trail_name"
        ].unique()
    )

    remaining_locations = (
        weather_locations[
            ~weather_locations[
                "trail_name"
            ]
            .isin(
                completed_trails
            )
        ]
        .copy()
    )

    print()
    print(
        f"Completed trails: "
        f"{len(completed_trails):,}"
    )

    print(
        f"Remaining trails: "
        f"{len(remaining_locations):,}"
    )

    if remaining_locations.empty:

        print()
        print(
            "All trails have already "
            "been downloaded."
        )

        all_weather = progress

    else:

        # -----------------------------------------------------
        # DOWNLOAD
        # -----------------------------------------------------

        print()
        print(
            f"Historical start date: "
            f"{START_DATE}"
        )

        print(
            f"Historical end date: "
            f"{end_date}"
        )

        total_batches = (
            len(
                remaining_locations
            )
            + BATCH_SIZE
            - 1
        ) // BATCH_SIZE

        print(
            f"Batch size: "
            f"{BATCH_SIZE}"
        )

        print(
            f"Remaining API batches: "
            f"{total_batches}"
        )

        print()

        all_weather_frames = []

        if not progress.empty:

            all_weather_frames.append(
                progress
            )

        for batch_number, batch in enumerate(
            chunk_dataframe(
                remaining_locations,
                BATCH_SIZE,
            ),
            start=1,
        ):

            first_name = (
                batch[
                    "trail_name"
                ]
                .iloc[0]
            )

            last_name = (
                batch[
                    "trail_name"
                ]
                .iloc[-1]
            )

            print(
                f"[Batch "
                f"{batch_number}/"
                f"{total_batches}] "
                f"{len(batch)} trails "
                f"({first_name} → "
                f"{last_name})"
            )

            response_data = (
                request_batch(
                    batch,
                    START_DATE,
                    end_date,
                )
            )

            responses = (
                normalize_response(
                    response_data,
                    len(batch),
                )
            )

            batch_frames = []

            for (
                (_, trail),
                location_response,
            ) in zip(
                batch.iterrows(),
                responses,
            ):

                trail_weather = (
                    parse_location_response(
                        trail,
                        location_response,
                    )
                )

                batch_frames.append(
                    trail_weather
                )

            # ---------------------------------------------
            # SAVE THIS BATCH IMMEDIATELY
            # ---------------------------------------------

            all_weather_frames.extend(
                batch_frames
            )

            save_progress(
                all_weather_frames,
                progress_path,
            )

            downloaded_count = sum(
                len(frame)
                for frame
                in batch_frames
            )

            print(
                f"  Saved "
                f"{downloaded_count:,} rows."
            )

            print(
                "  Progress safely written to disk."
            )

            if (
                batch_number
                < total_batches
            ):

                print(
                    f"  Waiting "
                    f"{SUCCESS_WAIT_SECONDS} "
                    "seconds before next batch..."
                )

                time.sleep(
                    SUCCESS_WAIT_SECONDS
                )

        all_weather = pd.concat(
            all_weather_frames,
            ignore_index=True,
        )

    # ---------------------------------------------------------
    # FINAL CLEANUP
    # ---------------------------------------------------------

    all_weather = (
        all_weather
        .drop_duplicates(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sort_values(
            [
                "trail_name",
                "date",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # ---------------------------------------------------------
    # VALIDATE FINAL DATASET
    # ---------------------------------------------------------

    duplicate_rows = (
        all_weather.duplicated(
            subset=[
                "trail_name",
                "date",
            ]
        )
        .sum()
    )

    unique_trails = (
        all_weather[
            "trail_name"
        ]
        .nunique()
    )

    unique_dates = (
        all_weather[
            "date"
        ]
        .nunique()
    )

    expected_dates = len(
        pd.date_range(
            START_DATE,
            end_date,
            freq="D",
        )
    )

    expected_rows = (
        len(
            weather_locations
        )
        * expected_dates
    )

    missing_trails = (
        set(
            weather_locations[
                "trail_name"
            ]
        )
        - set(
            all_weather[
                "trail_name"
            ]
        )
    )

    # ---------------------------------------------------------
    # SAVE FINAL DATASET
    # ---------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_weather.to_csv(
        output_path,
        index=False,
    )

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print(
        "MASTER HISTORICAL WEATHER DOWNLOAD COMPLETE"
    )
    print("=" * 72)

    print(
        f"Rows created: "
        f"{len(all_weather):,}"
    )

    print(
        f"Expected rows: "
        f"{expected_rows:,}"
    )

    print(
        f"Unique trails: "
        f"{unique_trails:,}"
    )

    print(
        f"Unique dates: "
        f"{unique_dates:,}"
    )

    print(
        f"Duplicate trail/date rows: "
        f"{duplicate_rows:,}"
    )

    print(
        f"Missing trails: "
        f"{len(missing_trails):,}"
    )

    if not all_weather.empty:

        print(
            f"First date: "
            f"{all_weather['date'].min().date()}"
        )

        print(
            f"Last date: "
            f"{all_weather['date'].max().date()}"
        )

    print()
    print(
        f"Saved final dataset to:"
        f"\n  {output_path}"
    )

    print()
    print(
        f"Resume/progress file remains at:"
        f"\n  {progress_path}"
    )


if __name__ == "__main__":
    main()