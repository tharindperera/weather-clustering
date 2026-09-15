from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import duckdb
import pandas as pd

from config import PROCESSED_DIR


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PARQUET_DIR = PROCESSED_DIR / "parquet" / "weather"

OUTPUT_DIR = PROCESSED_DIR / "features"
OUTPUT_FILE = OUTPUT_DIR / "clustering_input_original5.parquet"


# ---------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------

QUERY = f"""
SELECT
    location_id,
    city,
    city_ascii,
    country,
    iso2,
    iso3,

    AVG(temperature_mean) AS temperature_mean,
    AVG(precipitation_sum) AS precipitation_sum,
    AVG(relative_humidity_mean) AS relative_humidity_mean,
    AVG(wind_speed_mean) AS wind_speed_mean,
    AVG(surface_pressure_mean) AS surface_pressure_mean,

    COUNT(*) AS daily_observation_count

FROM read_parquet(
    '{PARQUET_DIR.as_posix()}/**/*.parquet',
    hive_partitioning = true
)

GROUP BY
    location_id,
    city,
    city_ascii,
    country,
    iso2,
    iso3

ORDER BY location_id
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BUILD ORIGINAL-VARIABLE CLUSTERING INPUT")
    print("=" * 80)

    con = duckdb.connect()

    try:
        df = con.execute(QUERY).df()
    finally:
        con.close()

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    expected_weather_columns = [
        "temperature_mean",
        "precipitation_sum",
        "relative_humidity_mean",
        "wind_speed_mean",
        "surface_pressure_mean",
    ]

    expected_id_columns = [
        "location_id",
        "city",
        "city_ascii",
        "country",
        "iso2",
        "iso3",
    ]

    expected_columns = (
        expected_id_columns
        + expected_weather_columns
        + ["daily_observation_count"]
    )

    missing_columns = [
        column for column in expected_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing expected columns: {missing_columns}"
        )

    if len(df) != 100:
        raise ValueError(
            f"Expected exactly 100 cities, found {len(df)}."
        )

    if df["location_id"].nunique() != 100:
        raise ValueError(
            "location_id values are not unique."
        )

    if df["city"].nunique() != 100:
        raise ValueError(
            "city values are not unique."
        )

    # Every city should have 3,653 daily observations:
    # 2016–2025 contains three leap years (2016, 2020, 2024).
    expected_daily_rows = 3653

    invalid_counts = df.loc[
        df["daily_observation_count"] != expected_daily_rows,
        ["city", "daily_observation_count"],
    ]

    if not invalid_counts.empty:
        raise ValueError(
            "Unexpected observation count for one or more cities:\n"
            f"{invalid_counts.to_string(index=False)}"
        )

    # Check missing and non-finite values.
    numeric_columns = expected_weather_columns

    if df[numeric_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in clustering variables."
        )

    if not df[numeric_columns].apply(
        lambda column: column.map(pd.notna).all()
    ).all():
        raise ValueError(
            "Invalid numeric values detected."
        )

    # ---------------------------------------------------------------
    # Display
    # ---------------------------------------------------------------

    print()
    print("Weather variables used for clustering:")
    for column in expected_weather_columns:
        print(f"  - {column}")

    print()
    print("Summary statistics:")
    print(
        df[expected_weather_columns]
        .describe()
        .round(3)
        .to_string()
    )

    print()
    print("Preview:")
    print(
        df[
            ["city", "country"] + expected_weather_columns
        ]
        .head(10)
        .round(3)
        .to_string(index=False)
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
    )

    print()
    print(f"Output written to:")
    print(f"  {OUTPUT_FILE}")

    print()
    print("=" * 80)
    print("VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
