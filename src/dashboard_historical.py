from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PROCESSED_DIR

# ---------------------------------------------------------------------
# Historical Parquet location
# ---------------------------------------------------------------------

HISTORICAL_PARQUET_DIR = PROCESSED_DIR / "parquet" / "weather"


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


import streamlit as st

@st.cache_data
def load_historical_data() -> pd.DataFrame:
    """
    Load the complete historical Parquet dataset.

    Returns
    -------
    pandas.DataFrame
        Historical city-day observations from 2016–2025.
    """

    if not HISTORICAL_PARQUET_DIR.exists():
        raise FileNotFoundError(
            f"Historical Parquet directory not found: {HISTORICAL_PARQUET_DIR}"
        )

    parquet_files = list(HISTORICAL_PARQUET_DIR.glob("**/*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {HISTORICAL_PARQUET_DIR}"
        )

    df = pd.read_parquet(
        HISTORICAL_PARQUET_DIR,
        engine="pyarrow",
    )

    required_columns = [
        "location_id",
        "city",
        "country",
        "date",
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "precipitation_sum",
        "relative_humidity_mean",
        "wind_speed_mean",
        "surface_pressure_mean",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing historical columns: {missing}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
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

