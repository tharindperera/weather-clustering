from __future__ import annotations

from typing import Any
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

CURRENT_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "surface_pressure",
    "weather_code",
    "is_day",
]


def geocode_city(
    city_name: str,
) -> dict[str, Any]:
    """
    Resolve a city name into an Open-Meteo location.
    Returns the best matching location.
    """
    city_name = city_name.strip()
    if not city_name:
        raise ValueError("City name cannot be empty.")

    params = {
        "name": city_name,
        "count": 5,
        "language": "en",
        "format": "json",
    }

    response = requests.get(
        GEOCODING_URL,
        params=params,
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    results = payload.get("results", [])
    if not results:
        raise ValueError(f"No location found for '{city_name}'.")

    # Open-Meteo ranks the returned matches.
    best = results[0]

    required_fields = [
        "name",
        "latitude",
        "longitude",
    ]

    missing = [field for field in required_fields if field not in best]

    if missing:
        raise ValueError(f"Geocoding result is missing: {missing}")

    return {
        "name": best["name"],
        "country": best.get("country", ""),
        "country_code": best.get("country_code", ""),
        "admin1": best.get("admin1", ""),
        "latitude": float(best["latitude"]),
        "longitude": float(best["longitude"]),
        "timezone": best.get("timezone", "auto"),
    }


def fetch_current_weather(
    latitude: float,
    longitude: float,
) -> dict[str, Any]:
    """
    Fetch current weather conditions from Open-Meteo.
    Only the current-weather payload is requested.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ",".join(CURRENT_VARIABLES),
        "timezone": "auto",
    }

    response = requests.get(
        FORECAST_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    if "current" not in payload:
        raise ValueError("Open-Meteo response does not contain current weather.")

    current = payload["current"]

    return {
        "time": current.get("time"),
        "temperature_c": current.get("temperature_2m"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "pressure_hpa": current.get("surface_pressure"),
        "weather_code": current.get("weather_code"),
        "is_day": current.get("is_day"),
        "timezone": payload.get("timezone", "auto"),
    }


def get_current_weather_for_city(
    city_name: str,
) -> dict[str, Any]:
    """
    Geocode a city and fetch its current weather.
    """
    location = geocode_city(city_name)
    weather = fetch_current_weather(
        location["latitude"],
        location["longitude"],
    )

    return {
        "location": location,
        "weather": weather,
    }
