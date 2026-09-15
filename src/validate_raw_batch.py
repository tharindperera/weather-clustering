import json
from pathlib import Path

from config import RAW_DIR


TEST_FILE = RAW_DIR / "test_batch_001.json"


def main() -> None:
    print("=" * 70)
    print("VALIDATING RAW OPEN-METEO BATCH")
    print("=" * 70)

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test batch not found: {TEST_FILE}"
        )

    with TEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    # -----------------------------------------------------
    # Open-Meteo returns a dictionary for one location
    # and a list for multiple locations.
    # -----------------------------------------------------

    if isinstance(data, list):
        results = data
    else:
        results = [data]

    print(f"\nLocations in response: {len(results)}")

    expected_daily_days = 366

    required_variables = [
        "time",
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
        "surface_pressure_mean",
    ]

    # -----------------------------------------------------
    # Validate each location
    # -----------------------------------------------------

    for index, location in enumerate(results, start=1):

        print(f"\n--- Location {index} ---")

        latitude = location.get("latitude")
        longitude = location.get("longitude")
        timezone = location.get("timezone")

        print(f"Latitude : {latitude}")
        print(f"Longitude: {longitude}")
        print(f"Timezone : {timezone}")

        if "daily" not in location:
            raise ValueError(
                f"Location {index} has no daily data."
            )

        daily = location["daily"]

        # Check variables.
        missing = [
            variable
            for variable in required_variables
            if variable not in daily
        ]

        if missing:
            raise ValueError(
                f"Location {index} is missing: {missing}"
            )

        # Check number of dates.
        date_count = len(daily["time"])

        print(f"Daily observations: {date_count}")

        if date_count != expected_daily_days:
            raise ValueError(
                f"Expected {expected_daily_days} days, "
                f"got {date_count}."
            )

        # Check every variable has the same length
        for variable in required_variables[1:]:
            value_count = len(daily[variable])

            if value_count != date_count:
                raise ValueError(
                    f"{variable} has {value_count} values "
                    f"but there are {date_count} dates."
                )

        # Show first and last date
        print(
            f"Date range: "
            f"{daily['time'][0]} → {daily['time'][-1]}"
        )

    print("\n" + "=" * 70)
    print("RAW BATCH VALIDATION PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()