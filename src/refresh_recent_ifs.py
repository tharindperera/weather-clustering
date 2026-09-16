from __future__ import annotations

import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

PARQUET_DIR = (
    DATA_DIR
    / "processed"
    / "parquet"
    / "weather_ifs_2026-present"
)

YEAR_PARQUET = (
    PARQUET_DIR
    / "year=2026"
    / "weather.parquet"
)

STATE_DIR = (
    DATA_DIR
    / "checkpoints"
    / "ifs_2026-present"
)

STATE_FILE = (
    STATE_DIR
    / "refresh_state.json"
)

# ============================================================================
# OPEN-METEO
# ============================================================================

API_URL = (
    "https://historical-forecast-api.open-meteo.com/v1/forecast"
)

MODEL = "ecmwf_ifs025"

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
# SETTINGS
# ============================================================================

BATCH_SIZE = 20
REQUEST_TIMEOUT_SECONDS = 120
REQUEST_DELAY_SECONDS = 5.0
MAX_RETRIES = 5

WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]

# ============================================================================
# DATE
# ============================================================================

def latest_complete_day() -> date:
    from datetime import datetime, timezone
    return (
        datetime.now(
            timezone.utc
        ).date()
        - timedelta(days=1)
    )

# ============================================================================
# LOCATIONS
# ============================================================================

def load_locations() -> pd.DataFrame:
    locations = pd.read_csv(
        LOCATIONS_FILE
    )

    required = [
        "location_id",
        "city",
        "city_ascii",
        "country",
        "iso2",
        "iso3",
        "admin_name",
        "capital",
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

    return locations[
        required
    ].copy()


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

# ============================================================================
# STATE
# ============================================================================

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {
            "last_successful_date": None,
            "updated_at": None,
        }
    return json.loads(
        STATE_FILE.read_text(
            encoding="utf-8"
        )
    )


def save_state(
    last_successful_date: date,
) -> None:
    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "last_successful_date": (
            last_successful_date.isoformat()
        ),
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }

    temporary = STATE_FILE.with_suffix(
        ".tmp"
    )

    temporary.write_text(
        json.dumps(
            state,
            indent=2,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        STATE_FILE
    )

# ============================================================================
# API
# ============================================================================

def fetch_batch(
    batch: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> list[dict]:
    params = {
        "latitude": ",".join(
            batch["latitude"].astype(str)
        ),
        "longitude": ",".join(
            batch["longitude"].astype(str)
        ),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": ",".join(
            DAILY_VARIABLES
        ),
        "timezone": "UTC",
        "models": MODEL,
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1,
    ):
        time.sleep(
            REQUEST_DELAY_SECONDS
        )

        try:
            response = requests.get(
                API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if response.status_code == 429:
                body = response.text.lower()
                if (
                    "hourly api request limit exceeded"
                    in body
                ):
                    raise RuntimeError(
                        "Open-Meteo hourly API limit exceeded."
                    )

                retry_after = response.headers.get(
                    "Retry-After"
                )

                if retry_after:
                    wait = min(
                        int(retry_after),
                        600,
                    )
                else:
                    wait = min(
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
                    f"429 received. "
                    f"Waiting {wait}s..."
                )
                time.sleep(wait)
                continue

            response.raise_for_status()

            payload = response.json()

            if not isinstance(
                payload,
                list,
            ):
                raise ValueError(
                    "Expected multi-location list response."
                )

            if len(payload) != len(batch):
                raise ValueError(
                    f"Expected {len(batch)} location responses, "
                    f"received {len(payload)}."
                )

            return payload

        except (
            requests.RequestException,
            ValueError,
        ) as exc:
            if attempt == MAX_RETRIES:
                raise

            wait = min(
                2 ** (
                    attempt - 1
                ),
                120,
            )

            print(
                f"Attempt {attempt} failed: {exc}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(
        "Request failed."
    )

# ============================================================================
# RESPONSE -> DATAFRAME
# ============================================================================

def responses_to_dataframe(
    batch: pd.DataFrame,
    responses: list[dict],
) -> pd.DataFrame:
    rows = []

    for location_index, response in enumerate(
        responses
    ):
        location = batch.iloc[
            location_index
        ]

        daily = response[
            "daily"
        ]

        dates = daily[
            "time"
        ]

        for i, date_value in enumerate(
            dates
        ):
            rows.append(
                {
                    "location_id": int(
                        location[
                            "location_id"
                        ]
                    ),
                    "city": location["city"],
                    "city_ascii": location[
                        "city_ascii"
                    ],
                    "country": location[
                        "country"
                    ],
                    "iso2": location["iso2"],
                    "iso3": location["iso3"],
                    "admin_name": location[
                        "admin_name"
                    ],
                    "capital": location[
                        "capital"
                    ],
                    "source_latitude": float(
                        location[
                            "latitude"
                        ]
                    ),
                    "source_longitude": float(
                        location[
                            "longitude"
                        ]
                    ),
                    "model_latitude": response.get(
                        "latitude"
                    ),
                    "model_longitude": response.get(
                        "longitude"
                    ),
                    "timezone": response.get(
                        "timezone",
                        "UTC",
                    ),
                    "date": date_value,
                    "temperature_mean": daily[
                        "temperature_2m_mean"
                    ][i],
                    "temperature_max": daily[
                        "temperature_2m_max"
                    ][i],
                    "temperature_min": daily[
                        "temperature_2m_min"
                    ][i],
                    "precipitation_sum": daily[
                        "precipitation_sum"
                    ][i],
                    "relative_humidity_mean": daily[
                        "relative_humidity_2m_mean"
                    ][i],
                    "wind_speed_mean": daily[
                        "wind_speed_10m_mean"
                    ][i],
                    "surface_pressure_mean": daily[
                        "surface_pressure_mean"
                    ][i],
                }
            )

    result = pd.DataFrame(rows)
    result["date"] = pd.to_datetime(
        result["date"]
    )
    result["year"] = (
        result["date"].dt.year
    )

    return result

# ============================================================================
# INCREMENTAL REFRESH LOGIC
# ============================================================================

def perform_incremental_refresh() -> dict:
    """
    Check for missing complete daily observations from Open-Meteo IFS,
    fetch missing records, and atomically update the Parquet dataset.
    """
    if not YEAR_PARQUET.exists():
        raise FileNotFoundError(
            "Existing recent IFS Parquet dataset not found."
        )

    locations = load_locations()

    current_df = pd.read_parquet(
        YEAR_PARQUET
    )

    current_df["date"] = pd.to_datetime(
        current_df["date"]
    )

    existing_max_date = (
        current_df["date"]
        .max()
        .date()
    )

    latest_day = latest_complete_day()

    # ---------------------------------------------------------------
    # Nothing new
    # ---------------------------------------------------------------
    if existing_max_date >= latest_day:
        save_state(existing_max_date)
        return {
            "status": "up_to_date",
            "existing_latest_date": existing_max_date,
            "latest_complete_date": latest_day,
            "new_rows": 0,
            "message": f"Dataset is already up to date through {existing_max_date}. No new complete daily observations are available yet.",
        }

    # ---------------------------------------------------------------
    # Missing date range
    # ---------------------------------------------------------------
    start_date = existing_max_date + timedelta(days=1)
    end_date = latest_day
    missing_days = (end_date - start_date).days + 1

    batches = make_batches(locations)
    new_frames = []

    for batch_number, batch in enumerate(batches, start=1):
        responses = fetch_batch(batch, start_date, end_date)
        frame = responses_to_dataframe(batch, responses)
        new_frames.append(frame)

    new_df = pd.concat(new_frames, ignore_index=True)
    expected_rows = len(locations) * missing_days

    if len(new_df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} new rows, received {len(new_df)}."
        )

    # ---------------------------------------------------------------
    # Upsert into existing Parquet
    # ---------------------------------------------------------------
    updated_df = pd.concat([current_df, new_df], ignore_index=True)
    updated_df = (
        updated_df.drop_duplicates(
            subset=["location_id", "date"], keep="last"
        )
        .sort_values(["location_id", "date"])
        .reset_index(drop=True)
    )

    expected_total_rows = (
        len(locations) * (latest_day - pd.Timestamp("2026-01-01").date()).days
        + len(locations)
    )

    if len(updated_df) != expected_total_rows:
        raise ValueError(
            f"Expected {expected_total_rows} total rows, received {len(updated_df)}."
        )

    # Quality checks
    duplicate_count = int(
        updated_df.duplicated(subset=["location_id", "date"]).sum()
    )
    if duplicate_count != 0:
        raise ValueError("Duplicate city-date rows detected.")

    missing_values = int(
        updated_df[WEATHER_COLUMNS].isna().sum().sum()
    )
    if missing_values != 0:
        raise ValueError(f"Missing weather values: {missing_values}")

    invalid_temperature = int(
        (
            (updated_df["temperature_min"] > updated_df["temperature_mean"])
            | (updated_df["temperature_mean"] > updated_df["temperature_max"])
        ).sum()
    )
    if invalid_temperature != 0:
        raise ValueError("Temperature consistency check failed.")

    # Atomic Parquet replacement
    temporary_file = YEAR_PARQUET.with_suffix(".parquet.tmp")
    updated_df.to_parquet(temporary_file, index=False, engine="pyarrow")
    temporary_file.replace(YEAR_PARQUET)
    save_state(latest_day)

    return {
        "status": "updated",
        "existing_latest_date": existing_max_date,
        "latest_complete_date": latest_day,
        "new_rows": len(new_df),
        "total_rows": len(updated_df),
        "message": f"Successfully updated dataset with {len(new_df):,} new observations through {latest_day}.",
    }

# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("INCREMENTAL ECMWF IFS REFRESH")
    print("=" * 80)

    result = perform_incremental_refresh()

    print()
    print(f"Existing latest date: {result['existing_latest_date']}")
    print(f"Latest complete date: {result['latest_complete_date']}")
    print()

    if result["status"] == "up_to_date":
        print("No new complete dates are available.")
        print("No Open-Meteo API request was made.")
        print()
        print("=" * 80)
        print("INCREMENTAL REFRESH COMPLETE — NOTHING TO UPDATE")
        print("=" * 80)
    else:
        print("-" * 80)
        print("REFRESH VALIDATION")
        print("-" * 80)
        print(f"New rows: {result['new_rows']:,}")
        print(f"Total rows: {result['total_rows']:,}")
        print(f"Latest date: {result['latest_complete_date']}")
        print()
        print("=" * 80)
        print("INCREMENTAL ECMWF IFS REFRESH COMPLETE")
        print("=" * 80)


if __name__ == "__main__":
    main()
