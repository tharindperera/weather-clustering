from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
if not (DATA_DIR / "locations" / "locations.csv").exists() and (PROJECT_ROOT / "weather-clustering" / "data" / "locations" / "locations.csv").exists():
    DATA_DIR = PROJECT_ROOT / "weather-clustering" / "data"

LOCATIONS_FILE = (
    DATA_DIR
    / "locations"
    / "locations.csv"
)

EXPERIMENT_NAME = "ifs_2026-present"

RAW_DIR = (
    DATA_DIR
    / "raw"
    / "historical"
    / EXPERIMENT_NAME
)

CHECKPOINT_DIR = (
    DATA_DIR
    / "checkpoints"
    / EXPERIMENT_NAME
)

MANIFEST_FILE = (
    CHECKPOINT_DIR
    / "historical_manifest.json"
)

BUDGET_FILE = (
    CHECKPOINT_DIR
    / "api_budget.json"
)


# ============================================================================
# OPEN-METEO
# ============================================================================

API_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

MODEL = "ecmwf_ifs025"


# ============================================================================
# DATE CONFIGURATION
# ============================================================================

START_DATE = date(
    2026,
    1,
    1,
)

WINDOW_DAYS = 14


def latest_complete_day() -> date:
    """
    Return the latest completed UTC calendar day.

    We deliberately do not ingest the current partial day because
    it is not a complete daily observation yet.
    """

    return (
        datetime.now(
            timezone.utc
        ).date()
        - timedelta(days=1)
    )


# ============================================================================
# WEATHER VARIABLES
# ============================================================================

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "surface_pressure_mean",
]


# ============================================================================
# REQUEST SAFETY
# ============================================================================

BATCH_SIZE = 20

REQUEST_DELAY_SECONDS = 5.0

MAX_RETRIES = 5

REQUEST_TIMEOUT_SECONDS = 120


# This is a project-level safety guard, not a guarantee of provider limits.
DAILY_API_SAFETY_LIMIT = 7000


# ============================================================================
# FILE / MANIFEST HELPERS
# ============================================================================

def ensure_directories() -> None:

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_json_file(
    path: Path,
    default: Any,
) -> Any:

    if not path.exists():
        return default

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def save_json_file(
    path: Path,
    payload: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    temporary.replace(
        path
    )


def load_manifest() -> dict:

    manifest = load_json_file(
        MANIFEST_FILE,
        {
            "experiment": EXPERIMENT_NAME,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "updated_at": None,
            "records": {},
        },
    )

    if "records" not in manifest:
        manifest["records"] = {}

    return manifest


def load_budget() -> dict:

    today = date.today().isoformat()

    budget = load_json_file(
        BUDGET_FILE,
        {
            "date": today,
            "requests": 0,
        },
    )

    # Reset local request counter on a new UTC day.
    if budget.get("date") != today:

        budget = {
            "date": today,
            "requests": 0,
        }

        save_json_file(
            BUDGET_FILE,
            budget,
        )

    return budget


def save_runtime_state(
    manifest: dict,
    budget: dict,
) -> None:

    manifest["updated_at"] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    save_json_file(
        MANIFEST_FILE,
        manifest,
    )

    save_json_file(
        BUDGET_FILE,
        budget,
    )


# ============================================================================
# LOCATION LOADING
# ============================================================================

def load_locations() -> pd.DataFrame:

    locations = pd.read_csv(
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
        if column not in locations.columns
    ]

    if missing:
        raise ValueError(
            f"Missing location columns: {missing}"
        )

    if len(locations) != 100:
        raise ValueError(
            f"Expected 100 locations, found {len(locations)}."
        )

    if locations["location_id"].nunique() != 100:
        raise ValueError(
            "Expected 100 unique location IDs."
        )

    return locations[
        required
    ].copy()


# ============================================================================
# WORK UNIT HELPERS
# ============================================================================

def make_batches(
    locations: pd.DataFrame,
) -> list[pd.DataFrame]:

    return [
        locations.iloc[
            start:start + BATCH_SIZE
        ].copy()
        for start in range(
            0,
            len(locations),
            BATCH_SIZE,
        )
    ]


def make_windows(
    start: date,
    end: date,
) -> list[tuple[date, date]]:

    windows = []

    current = start

    while current <= end:

        window_end = min(
            current
            + timedelta(
                days=WINDOW_DAYS - 1
            ),
            end,
        )

        windows.append(
            (
                current,
                window_end,
            )
        )

        current = (
            window_end
            + timedelta(days=1)
        )

    return windows


def work_unit_key(
    batch_number: int,
    start: date,
    end: date,
) -> str:

    return (
        f"batch_{batch_number:04d}"
        f"__{start.isoformat()}"
        f"__{end.isoformat()}"
    )


def raw_file_path(
    batch_number: int,
    start: date,
    end: date,
) -> Path:

    return (
        RAW_DIR
        / (
            f"batch_{batch_number:04d}"
            f"__{start.isoformat()}"
            f"__{end.isoformat()}"
            ".json"
        )
    )


# ============================================================================
# RESPONSE VALIDATION
# ============================================================================

def validate_location_response(
    response: dict,
    expected_dates: list[str],
) -> None:

    daily = response.get(
        "daily"
    )

    if not isinstance(
        daily,
        dict,
    ):
        raise ValueError(
            "Response does not contain a valid 'daily' object."
        )

    if "time" not in daily:
        raise ValueError(
            "Response does not contain daily dates."
        )

    returned_dates = daily[
        "time"
    ]

    if returned_dates != expected_dates:

        raise ValueError(
            "Returned daily dates do not exactly match "
            "the requested date window."
        )

    for variable in DAILY_VARIABLES:

        if variable not in daily:
            raise ValueError(
                f"Missing daily variable: {variable}"
            )

        values = daily[
            variable
        ]

        if len(values) != len(
            expected_dates
        ):
            raise ValueError(
                f"{variable}: expected "
                f"{len(expected_dates)} values, "
                f"received {len(values)}."
            )


def validate_batch_response(
    payload: Any,
    batch: pd.DataFrame,
    start: date,
    end: date,
) -> None:

    if not isinstance(
        payload,
        list,
    ):
        raise ValueError(
            "Expected Open-Meteo multi-location response to be a list."
        )

    if len(payload) != len(
        batch
    ):
        raise ValueError(
            f"Expected {len(batch)} location responses, "
            f"received {len(payload)}."
        )

    expected_dates = [
        (
            start
            + timedelta(days=offset)
        ).isoformat()
        for offset in range(
            (end - start).days + 1
        )
    ]

    for response in payload:

        validate_location_response(
            response,
            expected_dates,
        )


# ============================================================================
# API REQUEST
# ============================================================================

def request_batch(
    batch: pd.DataFrame,
    start: date,
    end: date,
    budget: dict,
) -> Any:

    if (
        budget["requests"]
        >= DAILY_API_SAFETY_LIMIT
    ):
        raise RuntimeError(
            "Local daily API safety limit reached. "
            "Stopping the ingestion run."
        )

    params = {
        "latitude": ",".join(
            batch["latitude"].astype(str)
        ),
        "longitude": ",".join(
            batch["longitude"].astype(str)
        ),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(
            DAILY_VARIABLES
        ),
        "timezone": "UTC",
        "models": MODEL,
    }

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):

        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(
                REQUEST_DELAY_SECONDS
            )

        try:

            response = requests.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            budget["requests"] += 1

            # ---------------------------------------------------------
            # Explicit rate-limit handling
            # ---------------------------------------------------------

            if response.status_code == 429:

                retry_after = response.headers.get(
                    "Retry-After"
                )

                body = response.text.lower()

                if (
                    "hourly api request limit exceeded"
                    in body
                ):

                    raise RuntimeError(
                        "Open-Meteo hourly API request limit exceeded. "
                        "Stopping immediately rather than retrying."
                    )

                if retry_after:

                    wait_seconds = min(
                        max(
                            int(
                                retry_after
                            ),
                            1,
                        ),
                        600,
                    )

                else:

                    wait_seconds = min(
                        60
                        * (
                            2
                            ** (
                                attempt
                                - 1
                            )
                        ),
                        600,
                    )

                print(
                    f"[429] Attempt {attempt}/{MAX_RETRIES}. "
                    f"Waiting {wait_seconds}s."
                )

                time.sleep(
                    wait_seconds
                )

                continue

            response.raise_for_status()

            payload = response.json()

            return payload

        except (
            requests.RequestException,
            ValueError,
        ) as exc:

            last_error = exc

            if attempt >= MAX_RETRIES:

                raise RuntimeError(
                    f"Request failed after "
                    f"{MAX_RETRIES} attempts: {exc}"
                ) from exc

            wait_seconds = min(
                2 ** (
                    attempt - 1
                ),
                120,
            )

            print(
                f"[Retry] Attempt {attempt}/{MAX_RETRIES} "
                f"failed: {exc}. "
                f"Waiting {wait_seconds}s."
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        f"Request failed: {last_error}"
    )


# ============================================================================
# MAIN INGESTION
# ============================================================================

def main() -> None:

    print("=" * 80)
    print(
        "RECENT ECMWF IFS INCREMENTAL INGESTION"
    )
    print("=" * 80)

    ensure_directories()

    locations = load_locations()

    manifest = load_manifest()
    budget = load_budget()

    latest_day = latest_complete_day()

    print()
    print(
        f"Latest complete UTC day: "
        f"{latest_day}"
    )

    if latest_day < START_DATE:

        print(
            "No complete IFS dates are available "
            "after the configured start date."
        )

        return

    batches = make_batches(
        locations
    )

    windows = make_windows(
        START_DATE,
        latest_day,
    )

    total_work_units = (
        len(batches)
        * len(windows)
    )

    successful = 0
    skipped = 0
    repaired = 0

    print()
    print(
        f"Cities: {len(locations)}"
    )

    print(
        f"Location batches: {len(batches)}"
    )

    print(
        f"Date windows: {len(windows)}"
    )

    print(
        f"Potential work units: "
        f"{total_work_units}"
    )

    # -----------------------------------------------------------------
    # Process every batch/window
    # -----------------------------------------------------------------

    for batch_index, batch in enumerate(
        batches,
        start=1,
    ):

        for start, end in windows:

            key = work_unit_key(
                batch_index,
                start,
                end,
            )

            raw_path = raw_file_path(
                batch_index,
                start,
                end,
            )

            record = manifest[
                "records"
            ].get(
                key
            )

            # ---------------------------------------------------------
            # Successful checkpoint + raw file exists
            # ---------------------------------------------------------

            if (
                record
                and record.get(
                    "status"
                )
                == "success"
                and raw_path.exists()
            ):

                skipped += 1

                continue

            # ---------------------------------------------------------
            # Checkpoint says success but raw file disappeared
            # ---------------------------------------------------------

            if (
                record
                and record.get(
                    "status"
                )
                == "success"
                and not raw_path.exists()
            ):

                print()
                print(
                    f"[RECOVER] Missing raw file: {key}"
                )

                repaired += 1

            print()
            print(
                f"[FETCH] {key}"
            )

            payload = request_batch(
                batch,
                start,
                end,
                budget,
            )

            validate_batch_response(
                payload,
                batch,
                start,
                end,
            )

            # ---------------------------------------------------------
            # Save raw API response
            # ---------------------------------------------------------

            temporary_path = raw_path.with_suffix(
                ".json.tmp"
            )

            with temporary_path.open(
                "w",
                encoding="utf-8",
            ) as handle:

                json.dump(
                    {
                        "experiment": EXPERIMENT_NAME,
                        "model": MODEL,
                        "batch_number": batch_index,
                        "start_date": start.isoformat(),
                        "end_date": end.isoformat(),
                        "location_ids": (
                            batch[
                                "location_id"
                            ]
                            .tolist()
                        ),
                        "locations": payload,
                    },
                    handle,
                    indent=2,
                    ensure_ascii=False,
                )

            temporary_path.replace(
                raw_path
            )

            # ---------------------------------------------------------
            # Checkpoint only after successful write + validation
            # ---------------------------------------------------------

            manifest[
                "records"
            ][key] = {
                "status": "success",
                "batch_number": batch_index,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "raw_file": str(
                    raw_path.relative_to(
                        PROJECT_ROOT
                    )
                ),
                "locations": len(
                    batch
                ),
                "completed_at": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

            save_runtime_state(
                manifest,
                budget,
            )

            successful += 1

            print(
                f"[SUCCESS] "
                f"{key}"
            )

    # -----------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------

    success_count = sum(
        1
        for record in manifest[
            "records"
        ].values()
        if record.get(
            "status"
        )
        == "success"
    )

    print()
    print("=" * 80)
    print(
        "RECENT IFS INGESTION SUMMARY"
    )
    print("=" * 80)

    print(
        f"Successful this run: {successful}"
    )

    print(
        f"Skipped existing: {skipped}"
    )

    print(
        f"Recovered missing raw files: {repaired}"
    )

    print(
        f"Total successful work units: "
        f"{success_count}"
    )

    print(
        f"Local requests used today: "
        f"{budget['requests']}"
    )

    print(
        f"Latest complete day: "
        f"{latest_day}"
    )

    print()
    print(
        "Historical ERA5 2016–2025 data "
        "and the K=4 clustering model were not modified."
    )

    print()
    print("=" * 80)
    print(
        "RECENT IFS INCREMENTAL INGESTION COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
