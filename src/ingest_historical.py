from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd

from budget import ApiBudget, BudgetExceededError
from checkpoint import CheckpointManager
from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    DAILY_API_SAFETY_LIMIT,
    DAILY_VARIABLES,
    HISTORICAL_API_URL,
    HISTORICAL_END_DATE,
    HISTORICAL_START_DATE,
    LOCATIONS_FILE,
    MAX_RETRIES,
    RAW_DIR,
    REQUEST_TIMEOUT_SECONDS,
    WINDOW_DAYS,
)
from openmeteo_client import OpenMeteoClient
from validator import validate_response


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def date_windows(
    start: date,
    end: date,
    window_days: int,
) -> list[tuple[date, date]]:
    """
    Generate inclusive date windows.

    Example:
        2024-01-01 → 2024-01-14
        2024-01-15 → 2024-01-28
        ...
    """

    windows: list[tuple[date, date]] = []

    current = start

    while current <= end:

        window_end = min(
            current + timedelta(days=window_days - 1),
            end,
        )

        windows.append(
            (current, window_end)
        )

        current = (
            window_end
            + timedelta(days=1)
        )

    return windows


def load_locations() -> pd.DataFrame:
    """Load the frozen location catalogue."""

    if not LOCATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Location file not found: {LOCATIONS_FILE}"
        )

    df = pd.read_csv(
        LOCATIONS_FILE
    )

    required = [
        "location_id",
        "city",
        "country",
        "latitude",
        "longitude",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing location columns: {missing}"
        )

    return df


def location_batches(
    locations: pd.DataFrame,
    batch_size: int,
):
    """Yield deterministic location batches."""

    for start in range(
        0,
        len(locations),
        batch_size,
    ):

        yield locations.iloc[
            start:start + batch_size
        ].copy()


def build_params(
    batch: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> dict:

    return {
        "latitude": ",".join(
            batch["latitude"].astype(str)
        ),
        "longitude": ",".join(
            batch["longitude"].astype(str)
        ),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "models": "era5",
        "daily": ",".join(
            DAILY_VARIABLES
        ),
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }


def work_id(
    batch_number: int,
    start_date: date,
    end_date: date,
) -> str:

    return (
        f"batch_{batch_number:04d}"
        f"__{start_date.isoformat()}"
        f"__{end_date.isoformat()}"
    )


def raw_file_path(
    work_id_value: str,
) -> Path:

    extraction_name = (
        f"era5_"
        f"{HISTORICAL_START_DATE}_"
        f"{HISTORICAL_END_DATE}"
    )

    return (
        RAW_DIR
        / extraction_name
        / f"{work_id_value}.json"
    )


# ---------------------------------------------------------
# Dry run
# ---------------------------------------------------------


def print_plan(
    locations: pd.DataFrame,
    batches: list[pd.DataFrame],
    windows: list[tuple[date, date]],
) -> None:

    total_work_units = (
        len(batches)
        * len(windows)
    )

    print("=" * 70)
    print("HISTORICAL INGESTION DRY RUN")
    print("=" * 70)

    print(
        f"\nLocations          : {len(locations)}"
    )

    print(
        f"Batch size         : {BATCH_SIZE}"
    )

    print(
        f"Location batches   : {len(batches)}"
    )

    print(
        f"Window size        : {WINDOW_DAYS} days"
    )

    print(
        f"Date range         : "
        f"{HISTORICAL_START_DATE} → "
        f"{HISTORICAL_END_DATE}"
    )

    print(
        f"Date windows       : {len(windows)}"
    )

    print(
        f"Total work units   : {total_work_units}"
    )

    print(
        "\nFirst five work units:"
    )

    counter = 0

    for batch_number, _batch in enumerate(
        batches,
        start=1,
    ):

        for start_date, end_date in windows:

            counter += 1

            if counter > 5:
                return

            print(
                f"  "
                f"{work_id(batch_number, start_date, end_date)}"
            )


# ---------------------------------------------------------
# Actual ingestion
# ---------------------------------------------------------


def run_ingestion(
    max_work_units: int | None = None,
) -> None:

    locations = load_locations()

    start = parse_date(
        HISTORICAL_START_DATE
    )

    end = parse_date(
        HISTORICAL_END_DATE
    )

    batches = list(
        location_batches(
            locations,
            BATCH_SIZE,
        )
    )

    windows = date_windows(
        start,
        end,
        WINDOW_DAYS,
    )

    checkpoint = CheckpointManager(
        CHECKPOINT_DIR
        / "historical_manifest.json"
    )

    budget = ApiBudget(
        budget_file=(
            CHECKPOINT_DIR
            / "api_budget.json"
        ),
        daily_limit=DAILY_API_SAFETY_LIMIT,
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_work_units = 0

    print("=" * 70)
    print("HISTORICAL INGESTION")
    print("=" * 70)

    print(
        f"Locations: {len(locations)}"
    )

    print(
        f"Batches: {len(batches)}"
    )

    print(
        f"Windows: {len(windows)}"
    )

    print(
        f"Total work units: "
        f"{len(batches) * len(windows)}"
    )

    with OpenMeteoClient() as client:

        for batch_number, batch in enumerate(
            batches,
            start=1,
        ):

            for start_date, end_date in windows:

                current_work_id = work_id(
                    batch_number,
                    start_date,
                    end_date,
                )

                # -----------------------------------------
                # Checkpoint / raw-file consistency
                # -----------------------------------------

                output_file = raw_file_path(
                    current_work_id
                )

                # Case 1:
                # Checkpoint says SUCCESS.
                #
                # Only skip if the corresponding raw file
                # still exists.
                if checkpoint.is_completed(
                    current_work_id
                ):

                    if output_file.is_file():

                        print(
                            f"\nSKIP "
                            f"{current_work_id} "
                            f"(checkpoint + raw file verified)"
                        )

                        continue

                    print(
                        f"\nWARNING: "
                        f"{current_work_id} is marked complete "
                        f"but the raw file is missing."
                    )

                    print(
                        "The work unit will be downloaded again."
                    )

                # Case 2:
                # No successful checkpoint, but a raw file
                # already exists.
                #
                # Validate it before making another API call.
                elif output_file.is_file():

                    print(
                        f"\nFOUND EXISTING RAW FILE: "
                        f"{current_work_id}"
                    )

                    try:

                        with output_file.open(
                            "r",
                            encoding="utf-8",
                        ) as file:

                            existing_data = json.load(
                                file
                            )

                        existing_results = validate_response(
                            data=existing_data,
                            expected_locations=len(
                                batch
                            ),
                            expected_days=(
                                end_date
                                - start_date
                            ).days
                            + 1,
                            expected_start=(
                                start_date.isoformat()
                            ),
                            expected_end=(
                                end_date.isoformat()
                            ),
                        )

                        checkpoint.mark_success(
                            current_work_id,
                            metadata={
                                "locations": len(
                                    existing_results
                                ),
                                "start_date": (
                                    start_date.isoformat()
                                ),
                                "end_date": (
                                    end_date.isoformat()
                                ),
                                "raw_file": str(
                                    output_file
                                ),
                                "recovered_from_raw": True,
                            },
                        )

                        print(
                            f"RECOVERED: "
                            f"{current_work_id} "
                            f"(valid raw file)"
                        )

                        continue

                    except Exception as exc:

                        print(
                            f"WARNING: Existing raw file "
                            f"failed validation for "
                            f"{current_work_id}."
                        )

                        print(
                            f"Validation error: {exc}"
                        )

                        print(
                            "The file will be replaced "
                            "by a fresh API response."
                        )

                        try:

                            output_file.unlink()

                        except OSError as delete_error:

                            raise RuntimeError(
                                "Could not remove invalid "
                                f"raw file: {output_file}"
                            ) from delete_error

                if (
                    max_work_units is not None
                    and processed_work_units >= max_work_units
                ):
                    print(
                        f"\nReached max work units: "
                        f"{max_work_units}"
                    )

                    return

                print("\n" + "-" * 70)

                print(
                    f"Processing: "
                    f"{current_work_id}"
                )

                print(
                    f"Locations: "
                    f"{batch['city_ascii'].tolist()}"
                )

                print(
                    f"Date range: "
                    f"{start_date} → {end_date}"
                )

                # -----------------------------------------
                # Budget check
                # -----------------------------------------

                try:

                    budget.reserve(1)

                except BudgetExceededError:

                    print(
                        "\nDaily API safety budget "
                        "reached."
                    )

                    print(
                        "Stopping cleanly. "
                        "Resume later."
                    )

                    return

                # -----------------------------------------
                # API request
                # -----------------------------------------

                params = build_params(
                    batch,
                    start_date,
                    end_date,
                )

                try:

                    data = (
                        client
                        .fetch_historical(
                            params
                        )
                    )

                    # -------------------------------------
                    # Validate
                    # -------------------------------------

                    results = validate_response(
                        data=data,
                        expected_locations=len(batch),
                        expected_days=(
                            end_date
                            - start_date
                        ).days
                        + 1,
                        expected_start=(
                            start_date.isoformat()
                        ),
                        expected_end=(
                            end_date.isoformat()
                        ),
                    )

                    # -------------------------------------
                    # Save
                    # -------------------------------------

                    output_file = raw_file_path(
                        current_work_id
                    )

                    output_file.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    with output_file.open(
                        "w",
                        encoding="utf-8",
                    ) as file:

                        json.dump(
                            data,
                            file,
                            indent=2,
                        )

                    # -------------------------------------
                    # Checkpoint SUCCESS
                    # -------------------------------------

                    checkpoint.mark_success(
                        current_work_id,
                        metadata={
                            "locations": len(
                                results
                            ),
                            "start_date": (
                                start_date.isoformat()
                            ),
                            "end_date": (
                                end_date.isoformat()
                            ),
                            "raw_file": str(
                                output_file
                            ),
                        },
                    )

                    budget.record_success()

                    processed_work_units += 1

                    print(
                        f"SUCCESS: "
                        f"{current_work_id}"
                    )

                except Exception as exc:

                    budget.record_failure()

                    checkpoint.mark_failure(
                        current_work_id,
                        error=str(exc),
                        metadata={
                            "start_date": (
                                start_date.isoformat()
                            ),
                            "end_date": (
                                end_date.isoformat()
                            ),
                        },
                    )

                    print(
                        f"FAILED: "
                        f"{current_work_id}"
                    )

                    print(
                        f"Error: {exc}"
                    )

                    # Stop rather than blindly continuing.
                    #
                    # The checkpoint means the next run
                    # will retry only this failed unit.
                    raise


# ---------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Safe Open-Meteo ERA5 historical ingestion."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the planned extraction without "
            "making API requests."
        ),
    )

    parser.add_argument(
        "--max-work-units",
        type=int,
        default=None,
        help=(
            "Process at most this many work units. "
            "Useful for controlled testing."
        ),
    )

    args = parser.parse_args()

    locations = load_locations()

    start = parse_date(
        HISTORICAL_START_DATE
    )

    end = parse_date(
        HISTORICAL_END_DATE
    )

    batches = list(
        location_batches(
            locations,
            BATCH_SIZE,
        )
    )

    windows = date_windows(
        start,
        end,
        WINDOW_DAYS,
    )

    if args.dry_run:

        print_plan(
            locations,
            batches,
            windows,
        )

        return

    run_ingestion(
        max_work_units=args.max_work_units
    )


if __name__ == "__main__":
    main()