from __future__ import annotations

from pathlib import Path

# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Locations file
LOCATIONS_FILE = DATA_DIR / "locations" / "locations.csv"
if not LOCATIONS_FILE.exists():
    LOCATIONS_FILE = (
        PROJECT_ROOT
        / "weather-clustering"
        / "data"
        / "locations"
        / "locations.csv"
    )

EXPECTED_LOCATION_COUNT = 100

# =========================================================
# Recent ECMWF IFS Configuration
# =========================================================

RECENT_API_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

RECENT_MODEL = "ecmwf_ifs025"

RECENT_RAW_DIR = DATA_DIR / "raw" / "recent_ifs"
RECENT_PARQUET_DIR = PROCESSED_DIR / "parquet" / "recent_ifs"
RECENT_CHECKPOINT_DIR = DATA_DIR / "checkpoints" / "recent_ifs"
RECENT_CHECKPOINT_FILE = RECENT_CHECKPOINT_DIR / "recent_manifest.json"

BATCH_SIZE = 20

# Daily weather variables matching the ERA5 historical schema
DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "surface_pressure_mean",
]
