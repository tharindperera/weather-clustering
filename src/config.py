from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

LOCATIONS_FILE = (
    DATA_DIR
    / "locations"
    / "locations.csv"
)

RAW_DIR = (
    DATA_DIR
    / "raw"
    / "historical"
)

CHECKPOINT_DIR = (
    DATA_DIR
    / "checkpoints"
)


# ---------------------------------------------------------
# Open-Meteo
# ---------------------------------------------------------

HISTORICAL_API_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


# ---------------------------------------------------------
# Historical period
# ---------------------------------------------------------

HISTORICAL_START_DATE = "2016-01-01"
HISTORICAL_END_DATE = "2025-12-31"


# ---------------------------------------------------------
# Variables
# ---------------------------------------------------------

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "surface_pressure_mean",
]

# ---------------------------------------------------------
# Ingestion configuration
# ---------------------------------------------------------

BATCH_SIZE = 20
WINDOW_DAYS = 14

# ---------------------------------------------------------
# API safety configuration
# ---------------------------------------------------------

# Minimum delay between successful requests.
REQUEST_DELAY_SECONDS = 5.0

# Maximum number of retries after temporary failures.
MAX_RETRIES = 5

# Base backoff delay.
BACKOFF_BASE_SECONDS = 2.0

# HTTP timeout.
REQUEST_TIMEOUT_SECONDS = 120

# ---------------------------------------------------------
# API usage safety budget
# ---------------------------------------------------------

# Open-Meteo currently documents a higher free daily limit.
# We intentionally operate below it for safety.
DAILY_API_SAFETY_LIMIT = 7000