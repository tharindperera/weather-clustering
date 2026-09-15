import json
import time
from pathlib import Path

import pandas as pd
import requests

from config import (
    BATCH_SIZE,
    DAILY_VARIABLES,
    HISTORICAL_API_URL,
    MAX_RETRIES,
    REQUEST_DELAY_SECONDS,
    REQUEST_TIMEOUT_SECONDS,
    HISTORICAL_START_DATE,
    HISTORICAL_END_DATE,
)


def load_test_locations() -> pd.DataFrame:
    """Load 20 locations for the batch-size test."""

    from config import LOCATIONS_FILE

    df = pd.read_csv(LOCATIONS_FILE)

    return df.head(20).copy()


def build_request_params(batch: pd.DataFrame) -> dict:
    """Build a single Open-Meteo request for a batch."""

    return {
        "latitude": ",".join(
            batch["latitude"].astype(str)
        ),
        "longitude": ",".join(
            batch["longitude"].astype(str)
        ),
        "start_date": HISTORICAL_START_DATE,
        "end_date": HISTORICAL_END_DATE,
        "models": "era5",
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }


def fetch_with_retry(
    session: requests.Session,
    params: dict,
) -> dict:

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                HISTORICAL_API_URL,
                params=params,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            print(
                f"HTTP {response.status_code} "
                f"(attempt {attempt})"
            )

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429:

                wait = 2 ** attempt

                print(
                    f"Rate limited. "
                    f"Waiting {wait} seconds..."
                )

                time.sleep(wait)

                continue

            response.raise_for_status()

        except requests.RequestException as exc:

            if attempt == MAX_RETRIES:
                raise

            wait = 2 ** attempt

            print(
                f"Temporary request error: {exc}"
            )

            print(
                f"Retrying in {wait} seconds..."
            )

            time.sleep(wait)

    raise RuntimeError(
        "Request failed after all retries."
    )


def main() -> None:

    print("=" * 70)
    print("HISTORICAL INGESTION TEST")
    print("=" * 70)

    locations = load_test_locations()
    print(
        f"Historical period: "
        f"{HISTORICAL_START_DATE} → {HISTORICAL_END_DATE}"
    )

    print(
        f"Testing with {len(locations)} locations."
    )

    print(
        locations[
            [
                "city",
                "country",
                "latitude",
                "longitude",
            ]
        ].to_string(index=False)
    )

    params = build_request_params(locations)

    print("\nRequesting Open-Meteo...")

    with requests.Session() as session:

        data = fetch_with_retry(
            session,
            params,
        )

    print("\nRequest successful.")

    # -----------------------------------------------------
    # Verify response structure
    # -----------------------------------------------------

    if isinstance(data, list):

        results = data

    else:

        results = [data]

    print(
        f"Locations returned: {len(results)}"
    )

    # -----------------------------------------------------
    # Save raw response
    # -----------------------------------------------------

    from config import RAW_DIR

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        RAW_DIR
        / "test_batch_001.json"
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

    print(
        f"\nRaw response saved to:\n{output_file}"
    )

    print("\nTest completed successfully.")


if __name__ == "__main__":
    main()