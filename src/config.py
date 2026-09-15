from pathlib import Path


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"


# =========================================================
# Active location catalogue
# =========================================================

LOCATIONS_FILE = (
    DATA_DIR
    / "locations"
    / "locations.csv"
)

EXPECTED_LOCATION_COUNT = 100


# =========================================================
# Experiment identity
# =========================================================

EXPERIMENT_NAME = (
    "era5_100cities_2016-01-01_2025-12-31"
)


# =========================================================
# Raw historical data
# =========================================================

RAW_DIR = (
    DATA_DIR
    / "raw"
    / "historical"
    / EXPERIMENT_NAME
)


# =========================================================
# Checkpoints and API budget
# =========================================================

CHECKPOINT_DIR = (
    DATA_DIR
    / "checkpoints"
    / EXPERIMENT_NAME
)

CHECKPOINT_FILE = (
    CHECKPOINT_DIR
    / "historical_manifest.json"
)

API_BUDGET_FILE = (
    CHECKPOINT_DIR
    / "api_budget.json"
)


# =========================================================
# Open-Meteo Historical API
# =========================================================

HISTORICAL_API_URL = (
    "https://archive-api.open-meteo.com/v1/archive"
)


# =========================================================
# Historical period
# =========================================================

HISTORICAL_START_DATE = "2016-01-01"
HISTORICAL_END_DATE = "2025-12-31"


# =========================================================
# Requested daily variables
# =========================================================

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "surface_pressure_mean",
]


# =========================================================
# Ingestion configuration
# =========================================================

BATCH_SIZE = 20
WINDOW_DAYS = 14


# =========================================================
# API request safety
# =========================================================

# Minimum delay between HTTP requests.
REQUEST_DELAY_SECONDS = 5.0

# Maximum retry attempts for retryable failures.
MAX_RETRIES = 5

# Base backoff used for retryable transient failures.
BACKOFF_BASE_SECONDS = 2.0

# HTTP request timeout.
REQUEST_TIMEOUT_SECONDS = 120


# =========================================================
# Local project safety budget
# =========================================================

# Deliberately below the provider's documented daily limit.
# This is our own conservative project safety ceiling.
DAILY_API_SAFETY_LIMIT = 7000