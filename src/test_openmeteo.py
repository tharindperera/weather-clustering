import requests


API_URL = "https://archive-api.open-meteo.com/v1/archive"

PARAMS = {
    # Colombo
    "latitude": 6.9271,
    "longitude": 79.8612,

    # Small test period
    "start_date": "2024-01-01",
    "end_date": "2024-01-07",

    # Explicitly use ERA5 for our historical test
    "models": "era5",

    # Daily variables needed for our project
    "daily": ",".join([
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
        "surface_pressure_mean",
    ]),

    # Units
    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",

    # Use the location's local timezone
    "timezone": "auto",
}


def main() -> None:
    print("Requesting Open-Meteo historical weather...")
    print(f"Location : Colombo, Sri Lanka")
    print(f"Period   : {PARAMS['start_date']} → {PARAMS['end_date']}")
    print(f"Model    : {PARAMS['models']}")

    response = requests.get(
        API_URL,
        params=PARAMS,
        timeout=60,
    )

    print(f"\nHTTP status: {response.status_code}")

    # Raise an exception for HTTP errors.
    response.raise_for_status()

    data = response.json()

    print("\n=== LOCATION INFORMATION ===")
    print("Latitude :", data["latitude"])
    print("Longitude:", data["longitude"])
    print("Timezone :", data["timezone"])
    print("Elevation:", data["elevation"])

    print("\n=== DAILY UNITS ===")
    for key, value in data["daily_units"].items():
        print(f"{key}: {value}")

    print("\n=== DAILY DATA ===")

    dates = data["daily"]["time"]

    for i, date in enumerate(dates):
        print(
            f"{date} | "
            f"Temp={data['daily']['temperature_2m_mean'][i]:.1f} °C | "
            f"Rain={data['daily']['precipitation_sum'][i]:.1f} mm | "
            f"Humidity={data['daily']['relative_humidity_2m_mean'][i]:.1f}% | "
            f"Wind={data['daily']['wind_speed_10m_mean'][i]:.1f} km/h | "
            f"Pressure={data['daily']['surface_pressure_mean'][i]:.1f} hPa"
        )


if __name__ == "__main__":
    main()