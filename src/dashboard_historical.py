from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
import streamlit as st

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------
# Hugging Face and Local Dataset Configuration
# ---------------------------------------------------------------------

HF_REPO_ID = (
    "tharinduperera/weather-clustering-data"
)

HF_ERA5_DASHBOARD_FILE = (
    "processed/dashboard/"
    "weather_era5_2016_2025.parquet"
)

LOCAL_ERA5_DASHBOARD_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard"
    / "weather_era5_2016_2025.parquet"
)
if not LOCAL_ERA5_DASHBOARD_FILE.exists() and (PROJECT_ROOT / "weather-clustering" / "data" / "processed" / "dashboard" / "weather_era5_2016_2025.parquet").exists():
    LOCAL_ERA5_DASHBOARD_FILE = (
        PROJECT_ROOT
        / "weather-clustering"
        / "data"
        / "processed"
        / "dashboard"
        / "weather_era5_2016_2025.parquet"
    )

# ---------------------------------------------------------------------
# Human-readable weather variables
# ---------------------------------------------------------------------

HISTORICAL_VARIABLES = {
    "Temperature — Mean": "temperature_mean",
    "Temperature — Maximum": "temperature_max",
    "Temperature — Minimum": "temperature_min",
    "Rainfall / Precipitation": "precipitation_sum",
    "Relative Humidity": "relative_humidity_mean",
    "Wind Speed": "wind_speed_mean",
    "Surface Pressure": "surface_pressure_mean",
}


VARIABLE_UNITS = {
    "temperature_mean": "°C",
    "temperature_max": "°C",
    "temperature_min": "°C",
    "precipitation_sum": "mm",
    "relative_humidity_mean": "%",
    "wind_speed_mean": "km/h",
    "surface_pressure_mean": "hPa",
}


@st.cache_data
def load_historical_data(
    allow_local_fallback: bool = True,
) -> pd.DataFrame:
    """
    Load the frozen ERA5 2016-2025 dashboard dataset.

    Primary source:
        Hugging Face Dataset repository.

    Optional local fallback:
        Consolidated local Parquet file.
    """

    try:

        downloaded_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_ERA5_DASHBOARD_FILE,
            repo_type="dataset",
        )

        df = pd.read_parquet(
            downloaded_file
        )

    except Exception as hf_error:

        if (
            allow_local_fallback
            and LOCAL_ERA5_DASHBOARD_FILE.exists()
        ):

            print(
                "WARNING: Hugging Face ERA5 download failed. "
                "Using local dashboard fallback."
            )

            print(
                f"Hugging Face error: {hf_error}"
            )

            df = pd.read_parquet(
                LOCAL_ERA5_DASHBOARD_FILE
            )

        else:

            raise RuntimeError(
                "Unable to load the ERA5 historical "
                "dashboard dataset from Hugging Face."
            ) from hf_error

    df["date"] = pd.to_datetime(
        df["date"]
    )

    if len(df) != 365_300:
        raise ValueError(
            f"Expected 365,300 ERA5 observations, "
            f"found {len(df)}."
        )

    if df["location_id"].nunique() != 100:
        raise ValueError(
            "Expected ERA5 data for exactly 100 locations."
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
            "Duplicate ERA5 city-date rows detected."
        )

    return df


def filter_historical_data(
    df: pd.DataFrame,
    city: str,
    start_date,
    end_date,
) -> pd.DataFrame:
    """
    Filter historical data for one city and date range.
    """

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    result = df[
        (df["city"] == city)
        & (df["date"] >= start)
        & (df["date"] <= end)
    ].copy()

    return result.sort_values(
        "date"
    ).reset_index(
        drop=True
    )


def summarize_variable(
    df: pd.DataFrame,
    variable: str,
) -> dict:
    """
    Generate basic descriptive statistics for a selected
    historical variable.
    """

    if df.empty:
        raise ValueError(
            "Cannot summarize an empty dataset."
        )

    series = df[variable]

    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "minimum": float(series.min()),
        "maximum": float(series.max()),
        "median": float(series.median()),
    }


def aggregate_monthly(
    df: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """
    Aggregate a selected historical variable by month.

    Precipitation is summed because it represents accumulated
    daily precipitation. All other weather indicators are averaged.
    """

    if df.empty:
        return pd.DataFrame()

    work = df[["date", variable]].copy()

    work["month"] = (
        work["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    if variable == "precipitation_sum":
        result = (
            work.groupby("month")[variable]
            .sum()
            .reset_index()
        )
    else:
        result = (
            work.groupby("month")[variable]
            .mean()
            .reset_index()
        )

    return result.sort_values("month").reset_index(drop=True)


def aggregate_yearly(
    df: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """
    Aggregate a selected historical variable by year.

    Precipitation is summed to obtain annual precipitation.
    Other weather indicators use annual mean values.
    """

    if df.empty:
        return pd.DataFrame()

    work = df[["date", variable]].copy()

    work["year"] = work["date"].dt.year

    if variable == "precipitation_sum":
        result = (
            work.groupby("year")[variable]
            .sum()
            .reset_index()
        )
    else:
        result = (
            work.groupby("year")[variable]
            .mean()
            .reset_index()
        )

    return result.sort_values("year").reset_index(drop=True)


def eda_summary(
    df: pd.DataFrame,
    variable: str,
) -> dict:
    """
    Generate a fuller EDA-style numerical summary.
    """

    if df.empty:
        raise ValueError(
            "Cannot summarize an empty dataset."
        )

    series = df[variable].dropna()

    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "minimum": float(series.min()),
        "q1": float(series.quantile(0.25)),
        "median": float(series.median()),
        "q3": float(series.quantile(0.75)),
        "maximum": float(series.max()),
        "range": float(series.max() - series.min()),
    }
