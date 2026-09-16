import sys
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT / "weather-clustering" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "weather-clustering" / "src"))

try:
    from src.config_recent_ifs import (
        RECENT_API_URL,
        RECENT_MODEL,
        LOCATIONS_FILE,
        DAILY_VARIABLES,
        BATCH_SIZE,
    )
except ImportError:
    from config_recent_ifs import (
        RECENT_API_URL,
        RECENT_MODEL,
        LOCATIONS_FILE,
        DAILY_VARIABLES,
        BATCH_SIZE,
    )


def test_recent_ifs_batch() -> None:
    print("=== ECMWF IFS RECENT DATA BATCH API TEST ===")

    # 1. Load locations
    if not LOCATIONS_FILE.exists():
        raise FileNotFoundError(f"Locations file not found at {LOCATIONS_FILE}")

    locations_df = pd.read_csv(LOCATIONS_FILE)
    test_batch = locations_df.head(BATCH_SIZE).copy()
    print(f"Loaded {len(test_batch)} test locations from {LOCATIONS_FILE.name}")

    # 2. Compute 7-day test window (past 7 completed days)
    end_dt = date.today() - timedelta(days=2)
    start_dt = end_dt - timedelta(days=6)
    start_date_str = start_dt.strftime("%Y-%m-%d")
    end_date_str = end_dt.strftime("%Y-%m-%d")

    print(f"Test period: {start_date_str} to {end_date_str} (7 days)")
    print(f"Model: {RECENT_MODEL}")
    print(f"Variables: {', '.join(DAILY_VARIABLES)}")

    # 3. Format batch request
    latitudes = ",".join(test_batch["latitude"].astype(str))
    longitudes = ",".join(test_batch["longitude"].astype(str))

    params = {
        "latitude": latitudes,
        "longitude": longitudes,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "models": RECENT_MODEL,
        "daily": ",".join(DAILY_VARIABLES),
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "timezone": "auto",
    }

    print(f"\nSending batch GET request to {RECENT_API_URL}...")
    response = requests.get(RECENT_API_URL, params=params, timeout=60)
    print(f"HTTP Status: {response.status_code}")
    response.raise_for_status()

    payload = response.json()
    if not isinstance(payload, list):
        payload = [payload]

    print(f"Received responses for {len(payload)} locations (Expected: {BATCH_SIZE})")
    assert len(payload) == len(test_batch), (
        f"Expected {len(test_batch)} responses, got {len(payload)}"
    )

    # 4. Verify each city's daily records
    sample_rows = []
    total_observations = 0

    for i, (_, loc_row) in enumerate(test_batch.iterrows()):
        city_res = payload[i]
        daily = city_res.get("daily", {})
        dates = daily.get("time", [])
        total_observations += len(dates)

        # Verify required daily fields
        for var in DAILY_VARIABLES:
            assert var in daily, f"Missing {var} in response for {loc_row['city']}"
            assert len(daily[var]) == len(dates), (
                f"Length mismatch for {var} in {loc_row['city']}"
            )

        # Capture first observation for verification display
        if len(dates) > 0:
            sample_rows.append(
                {
                    "city": loc_row["city"],
                    "country": loc_row["country"],
                    "date": dates[0],
                    "temp_mean": daily["temperature_2m_mean"][0],
                    "precip": daily["precipitation_sum"][0],
                    "humidity": daily["relative_humidity_2m_mean"][0],
                    "wind": daily["wind_speed_10m_mean"][0],
                    "pressure": daily["surface_pressure_mean"][0],
                }
            )

    sample_df = pd.DataFrame(sample_rows)
    print(
        f"\nTotal daily observations received across {len(payload)} cities: "
        f"{total_observations}"
    )
    print("\nSample first-day observations across test cities:")
    print(sample_df.head(10).to_string(index=False))

    print("\n[SUCCESS] Phase 10A.1 / 10A.2 ECMWF IFS 20-city batch API test PASSED.")


if __name__ == "__main__":
    test_recent_ifs_batch()
