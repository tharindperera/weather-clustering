import sys
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


PARAMS = {
    # Colombo
    "latitude": 6.9271,
    "longitude": 79.8612,

    # Small recent test period
    "start_date": "2026-09-13",
    "end_date": "2026-09-15",

    # ECMWF IFS
    "models": "ecmwf_ifs025",

    # Hourly weather variables
    "hourly": ",".join([
        "temperature_2m",
        "relative_humidity_2m",
        "precipitation",
        "wind_speed_10m",
        "surface_pressure",
    ]),

    "temperature_unit": "celsius",
    "wind_speed_unit": "kmh",
    "precipitation_unit": "mm",

    "timezone": "auto",
}


def main() -> None:
    print("Requesting ECMWF IFS historical/recent weather...")
    print("Location : Colombo, Sri Lanka")
    print(
        f"Period   : "
        f"{PARAMS['start_date']} → {PARAMS['end_date']}"
    )
    print(f"Model    : {PARAMS['models']}")

    response = requests.get(
        API_URL,
        params=PARAMS,
        timeout=60,
    )

    print(f"\nHTTP status: {response.status_code}")

    response.raise_for_status()

    data = response.json()

    print("\n=== LOCATION INFORMATION ===")
    print("Latitude :", data["latitude"])
    print("Longitude:", data["longitude"])
    print("Timezone :", data["timezone"])

    print("\n=== HOURLY UNITS ===")
    for key, value in data["hourly_units"].items():
        print(f"{key}: {value}")

    print("\n=== FIRST 10 HOURLY RECORDS ===")

    times = data["hourly"]["time"]

    for i in range(min(10, len(times))):
        print(
            f"{times[i]} | "
            f"Temp={data['hourly']['temperature_2m'][i]:.1f} °C | "
            f"Humidity={data['hourly']['relative_humidity_2m'][i]:.1f}% | "
            f"Rain={data['hourly']['precipitation'][i]:.1f} mm | "
            f"Wind={data['hourly']['wind_speed_10m'][i]:.1f} km/h | "
            f"Pressure={data['hourly']['surface_pressure'][i]:.1f} hPa"
        )

    print("\nTotal hourly observations:", len(times))


if __name__ == "__main__":
    main()