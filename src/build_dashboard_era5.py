from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Support the current workspace layout while keeping the normal repo path first.
SOURCE_CANDIDATES = [
    PROJECT_ROOT
    / "data"
    / "processed"
    / "parquet"
    / "weather",

    PROJECT_ROOT
    / "weather-clustering"
    / "data"
    / "processed"
    / "parquet"
    / "weather",
]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard"
    / "weather_era5_2016_2025.parquet"
)

WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]


def find_source_directory() -> Path:

    for candidate in SOURCE_CANDIDATES:

        if (
            candidate.exists()
            and any(
                candidate.rglob("*.parquet")
            )
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find the ERA5 partitioned Parquet dataset."
    )


def main() -> None:

    print("=" * 80)
    print("BUILD CONSOLIDATED ERA5 DASHBOARD DATASET")
    print("=" * 80)

    source_dir = find_source_directory()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_files = list(
        source_dir.rglob("*.parquet")
    )

    print()
    print(
        f"Source directory: {source_dir}"
    )

    print(
        f"Source Parquet files: {len(parquet_files):,}"
    )

    if len(parquet_files) != 1350:
        raise ValueError(
            f"Expected 1,350 ERA5 Parquet files, "
            f"found {len(parquet_files)}."
        )

    source_glob = (
        source_dir
        / "**"
        / "*.parquet"
    ).as_posix()

    output_path = OUTPUT_FILE.as_posix()

    # ---------------------------------------------------------------
    # Consolidate efficiently with DuckDB.
    # Original partitioned files are read-only and remain untouched.
    # ---------------------------------------------------------------

    connection = duckdb.connect()

    connection.execute(
        f"""
        COPY (
            SELECT *
            FROM read_parquet(
                '{source_glob}',
                union_by_name = true
            )
        )
        TO '{output_path}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    connection.close()

    # ---------------------------------------------------------------
    # Validate consolidated output
    # ---------------------------------------------------------------

    df = pd.read_parquet(
        OUTPUT_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    print()
    print("-" * 80)
    print("VALIDATION")
    print("-" * 80)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Locations: {df['location_id'].nunique()}"
    )

    print(
        f"Cities: {df['city'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    if len(df) != 365_300:
        raise ValueError(
            f"Expected 365,300 rows, found {len(df)}."
        )

    if df["location_id"].nunique() != 100:
        raise ValueError(
            "Expected exactly 100 locations."
        )

    if (
        df["date"].min().date()
        != pd.Timestamp("2016-01-01").date()
    ):
        raise ValueError(
            "Unexpected ERA5 minimum date."
        )

    if (
        df["date"].max().date()
        != pd.Timestamp("2025-12-31").date()
    ):
        raise ValueError(
            "Unexpected ERA5 maximum date."
        )

    duplicates = int(
        df.duplicated(
            subset=[
                "location_id",
                "date",
            ]
        ).sum()
    )

    missing = int(
        df[
            WEATHER_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    invalid_temperature = int(
        (
            (
                df["temperature_min"]
                > df["temperature_mean"]
            )
            |
            (
                df["temperature_mean"]
                > df["temperature_max"]
            )
        ).sum()
    )

    print(
        f"Duplicate city-date rows: {duplicates}"
    )

    print(
        f"Missing weather values: {missing}"
    )

    print(
        "Temperature consistency violations: "
        f"{invalid_temperature}"
    )

    if duplicates != 0:
        raise ValueError(
            "Duplicate ERA5 city-date records detected."
        )

    if missing != 0:
        raise ValueError(
            "Missing ERA5 weather values detected."
        )

    if invalid_temperature != 0:
        raise ValueError(
            "ERA5 temperature consistency check failed."
        )

    print()
    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Output size: "
        f"{OUTPUT_FILE.stat().st_size / (1024 ** 2):.2f} MB"
    )

    print()
    print("=" * 80)
    print("ERA5 DASHBOARD DATASET COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
