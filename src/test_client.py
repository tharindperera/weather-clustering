import pandas as pd

from config import (
    DAILY_VARIABLES,
    HISTORICAL_END_DATE,
    HISTORICAL_START_DATE,
)
from openmeteo_client import OpenMeteoClient


def build_test_params() -> dict:
    """
    Build the same 20-city / 14-day request
    that we already know works.
    """

    from config import LOCATIONS_FILE

    locations = pd.read_csv(
        LOCATIONS_FILE
    ).head(20)

    return {
        "latitude": ",".join(
            locations["latitude"].astype(str)
        ),
        "longitude": ",".join(
            locations["longitude"].astype(str)
        ),
        "start_date": HISTORICAL_START_DATE,
        "end_date": HISTORICAL_END_DATE,
        "models": "era5",
        "daily": ",".join(
            DAILY_VARIABLES
        ),
        "timezone": "auto",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }


def main() -> None:

    print("=" * 70)
    print("OPEN-METEO CLIENT TEST")
    print("=" * 70)

    params = build_test_params()

    print(
        f"\nPeriod: "
        f"{HISTORICAL_START_DATE} → "
        f"{HISTORICAL_END_DATE}"
    )

    print("Locations: 20")
    print("Model: era5")

    with OpenMeteoClient() as client:

        data = client.fetch_historical(
            params
        )

    # -----------------------------------------------------
    # Basic response check
    # -----------------------------------------------------

    if isinstance(data, list):
        location_count = len(data)
    else:
        location_count = 1

    print(
        f"\nLocations returned: "
        f"{location_count}"
    )

    if location_count != 20:
        raise RuntimeError(
            f"Expected 20 locations, "
            f"received {location_count}."
        )

    print(
        "\nOPEN-METEO CLIENT TEST PASSED"
    )


if __name__ == "__main__":
    main()