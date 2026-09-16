from pathlib import Path
import pandas as pd

try:
    from src.refresh_recent_ifs import perform_incremental_refresh
except ImportError:
    from refresh_recent_ifs import perform_incremental_refresh

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
if not (DATA_DIR / "processed" / "parquet" / "weather_ifs_2026-present" / "year=2026" / "weather.parquet").exists() and (PROJECT_ROOT / "weather-clustering" / "data" / "processed" / "parquet" / "weather_ifs_2026-present" / "year=2026" / "weather.parquet").exists():
    DATA_DIR = PROJECT_ROOT / "weather-clustering" / "data"

RECENT_PARQUET = (
    DATA_DIR
    / "processed"
    / "parquet"
    / "weather_ifs_2026-present"
    / "year=2026"
    / "weather.parquet"
)

RECENT_WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]


def load_recent_ifs_data() -> pd.DataFrame:
    """
    Load the latest available ECMWF IFS recent-weather dataset.
    Falls back to Hugging Face if not found locally.
    """
    if not RECENT_PARQUET.exists():
        try:
            from huggingface_hub import hf_hub_download
            import shutil
            downloaded = hf_hub_download(
                repo_id="tharinduperera/weather-clustering-data",
                filename="processed/parquet/weather_ifs_2026-present/year=2026/weather.parquet",
                repo_type="dataset",
            )
            RECENT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(downloaded, RECENT_PARQUET)
        except Exception:
            pass

    if not RECENT_PARQUET.exists():
        raise FileNotFoundError(
            f"Recent IFS dataset not found: {RECENT_PARQUET}"
        )

    df = pd.read_parquet(
        RECENT_PARQUET
    )

    required_columns = [
        "location_id",
        "city",
        "country",
        "date",
        *RECENT_WEATHER_COLUMNS,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing recent IFS columns: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


def get_latest_recent_date(
    df: pd.DataFrame,
):
    return df["date"].max().date()


def get_recent_city_data(
    df: pd.DataFrame,
    city: str,
) -> pd.DataFrame:

    return (
        df[
            df["city"] == city
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )
