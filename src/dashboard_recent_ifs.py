from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


# ============================================================================
# HUGGING FACE DATASET
# ============================================================================

HF_REPO_ID = "tharinduperera/weather-clustering-data"

HF_PARQUET_PATH = (
    "processed/parquet/"
    "weather_ifs_2026-present/"
    "year=2026/"
    "weather.parquet"
)


# ============================================================================
# OPTIONAL LOCAL FALLBACK
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCAL_PARQUET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "parquet"
    / "weather_ifs_2026-present"
    / "year=2026"
    / "weather.parquet"
)
if not LOCAL_PARQUET.exists() and (PROJECT_ROOT / "weather-clustering" / "data" / "processed" / "parquet" / "weather_ifs_2026-present" / "year=2026" / "weather.parquet").exists():
    LOCAL_PARQUET = (
        PROJECT_ROOT
        / "weather-clustering"
        / "data"
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


REQUIRED_COLUMNS = [
    "location_id",
    "city",
    "country",
    "date",
    *RECENT_WEATHER_COLUMNS,
]


# ============================================================================
# VALIDATION
# ============================================================================

def validate_recent_ifs_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalize the recent ECMWF IFS analytical dataset.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Recent IFS dataset is missing required columns: "
            f"{missing_columns}"
        )

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "location_id",
                "date",
            ]
        ).sum()
    )

    if duplicate_count != 0:
        raise ValueError(
            f"Recent IFS dataset contains "
            f"{duplicate_count} duplicate city-date rows."
        )

    missing_weather_values = int(
        df[
            RECENT_WEATHER_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    if missing_weather_values != 0:
        raise ValueError(
            "Recent IFS dataset contains "
            f"{missing_weather_values} missing weather values."
        )

    if df["location_id"].nunique() != 100:
        raise ValueError(
            "Expected recent IFS data for exactly 100 locations, "
            f"found {df['location_id'].nunique()}."
        )

    return df


# ============================================================================
# HUGGING FACE LOADER
# ============================================================================

def download_recent_ifs_from_huggingface() -> Path:
    """
    Download the latest published ECMWF IFS Parquet file from
    the Hugging Face Dataset repository.

    Hugging Face handles its own local cache. If the remote file has
    changed, the latest version is retrieved automatically.
    """

    downloaded_file = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PARQUET_PATH,
        repo_type="dataset",
    )

    return Path(
        downloaded_file
    )


def load_recent_ifs_data(
    allow_local_fallback: bool = True,
) -> pd.DataFrame:
    """
    Load recent ECMWF IFS data.

    Primary source:
        Hugging Face Dataset repository.

    Optional fallback:
        Existing local Parquet file for local development if Hugging Face
        is temporarily unreachable.
    """

    try:

        parquet_file = (
            download_recent_ifs_from_huggingface()
        )

        df = pd.read_parquet(
            parquet_file
        )

        df = validate_recent_ifs_data(
            df
        )

        return df

    except Exception as hf_error:

        if (
            allow_local_fallback
            and LOCAL_PARQUET.exists()
        ):

            print(
                "WARNING: Hugging Face recent-data download failed. "
                "Using local fallback."
            )

            print(
                f"Hugging Face error: {hf_error}"
            )

            df = pd.read_parquet(
                LOCAL_PARQUET
            )

            df = validate_recent_ifs_data(
                df
            )

            return df

        raise RuntimeError(
            "Unable to load recent ECMWF IFS data "
            "from Hugging Face."
        ) from hf_error


# ============================================================================
# DASHBOARD HELPERS
# ============================================================================

def get_latest_recent_date(
    df: pd.DataFrame,
):
    return (
        df["date"]
        .max()
        .date()
    )


def get_recent_city_data(
    df: pd.DataFrame,
    city: str,
) -> pd.DataFrame:

    return (
        df[
            df["city"] == city
        ]
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )
