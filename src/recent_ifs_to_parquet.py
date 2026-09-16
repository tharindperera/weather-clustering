import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
if not (DATA_DIR / "locations" / "locations.csv").exists() and (PROJECT_ROOT / "weather-clustering" / "data" / "locations" / "locations.csv").exists():
    DATA_DIR = PROJECT_ROOT / "weather-clustering" / "data"

RAW_DIR = (
    DATA_DIR
    / "raw"
    / "historical"
    / "ifs_2026-present"
)

OUTPUT_DIR = (
    DATA_DIR
    / "processed"
    / "parquet"
    / "weather_ifs_2026-present"
)

LOCATIONS_FILE = (
    DATA_DIR
    / "locations"
    / "locations.csv"
)


DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "relative_humidity_2m_mean",
    "wind_speed_10m_mean",
    "surface_pressure_mean",
]


OUTPUT_COLUMNS = [
    "location_id",
    "city",
    "city_ascii",
    "country",
    "iso2",
    "iso3",
    "admin_name",
    "capital",
    "source_latitude",
    "source_longitude",
    "model_latitude",
    "model_longitude",
    "timezone",
    "date",
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
    "year",
]


def load_locations() -> pd.DataFrame:
    locations = pd.read_csv(
        LOCATIONS_FILE
    )

    required = [
        "location_id",
        "city",
        "city_ascii",
        "country",
        "iso2",
        "iso3",
        "admin_name",
        "capital",
        "latitude",
        "longitude",
    ]

    missing = [
        column
        for column in required
        if column not in locations.columns
    ]

    if missing:
        raise ValueError(
            f"Missing location columns: {missing}"
        )

    if len(locations) != 100:
        raise ValueError(
            f"Expected 100 locations, found {len(locations)}."
        )

    return locations[
        required
    ].copy()


def convert_file(
    raw_file: Path,
    locations_by_id: dict,
) -> list[dict]:

    with raw_file.open(
        "r",
        encoding="utf-8",
    ) as handle:

        payload = json.load(handle)

    location_ids = payload.get(
        "location_ids",
        [],
    )

    responses = payload.get(
        "locations",
        [],
    )

    if len(location_ids) != len(responses):
        raise ValueError(
            f"{raw_file.name}: location ID count does not match "
            f"response count."
        )

    rows = []

    for location_id, response in zip(
        location_ids,
        responses,
    ):

        location = locations_by_id.get(
            int(location_id)
        )

        if location is None:
            raise ValueError(
                f"{raw_file.name}: unknown location_id "
                f"{location_id}."
            )

        daily = response.get(
            "daily"
        )

        if daily is None:
            raise ValueError(
                f"{raw_file.name}: missing daily object."
            )

        dates = daily.get(
            "time",
            [],
        )

        expected_length = len(dates)

        for variable in DAILY_VARIABLES:

            if variable not in daily:
                raise ValueError(
                    f"{raw_file.name}: missing {variable}."
                )

            if len(daily[variable]) != expected_length:
                raise ValueError(
                    f"{raw_file.name}: {variable} length does "
                    f"not match dates."
                )

        for index, date_value in enumerate(
            dates
        ):

            rows.append(
                {
                    "location_id": int(
                        location_id
                    ),
                    "city": location["city"],
                    "city_ascii": location["city_ascii"],
                    "country": location["country"],
                    "iso2": location["iso2"],
                    "iso3": location["iso3"],
                    "admin_name": location["admin_name"],
                    "capital": location["capital"],
                    "source_latitude": float(
                        location["latitude"]
                    ),
                    "source_longitude": float(
                        location["longitude"]
                    ),
                    "model_latitude": response.get(
                        "latitude"
                    ),
                    "model_longitude": response.get(
                        "longitude"
                    ),
                    "timezone": response.get(
                        "timezone",
                        "UTC",
                    ),
                    "date": date_value,
                    "temperature_mean": daily[
                        "temperature_2m_mean"
                    ][index],
                    "temperature_max": daily[
                        "temperature_2m_max"
                    ][index],
                    "temperature_min": daily[
                        "temperature_2m_min"
                    ][index],
                    "precipitation_sum": daily[
                        "precipitation_sum"
                    ][index],
                    "relative_humidity_mean": daily[
                        "relative_humidity_2m_mean"
                    ][index],
                    "wind_speed_mean": daily[
                        "wind_speed_10m_mean"
                    ][index],
                    "surface_pressure_mean": daily[
                        "surface_pressure_mean"
                    ][index],
                    "year": int(
                        date_value[:4]
                    ),
                }
            )

    return rows


def main() -> None:

    print("=" * 80)
    print(
        "CONVERT RECENT ECMWF IFS JSON TO PARQUET"
    )
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    locations = load_locations()

    locations_by_id = {
        int(row["location_id"]): row
        for _, row in locations.iterrows()
    }

    raw_files = sorted(
        RAW_DIR.glob(
            "*.json"
        )
    )

    if len(raw_files) != 95:
        raise ValueError(
            f"Expected 95 raw JSON files, found {len(raw_files)}."
        )

    all_rows = []

    for index, raw_file in enumerate(
        raw_files,
        start=1,
    ):

        print(
            f"[{index:03d}/"
            f"{len(raw_files):03d}] "
            f"{raw_file.name}"
        )

        rows = convert_file(
            raw_file,
            locations_by_id,
        )

        all_rows.extend(
            rows
        )

    df = pd.DataFrame(
        all_rows,
        columns=OUTPUT_COLUMNS,
    )

    # -----------------------------------------------------------------
    # Normalize types
    # -----------------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )

    numeric_columns = [
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "precipitation_sum",
        "relative_humidity_mean",
        "wind_speed_mean",
        "surface_pressure_mean",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    print()
    print("-" * 80)
    print("VALIDATION")
    print("-" * 80)

    print(
        f"Rows: {len(df):,}"
    )

    if len(df) != 25_800:
        raise ValueError(
            f"Expected exactly 25,800 rows, found {len(df)}."
        )

    if df["location_id"].nunique() != 100:
        raise ValueError(
            "Expected 100 unique locations."
        )

    if df["city"].nunique() != 100:
        raise ValueError(
            "Expected 100 unique cities."
        )

    if df["date"].min().date() != pd.Timestamp(
        "2026-01-01"
    ).date():

        raise ValueError(
            "Unexpected minimum date."
        )

    if df["date"].max().date() != pd.Timestamp(
        "2026-09-15"
    ).date():

        raise ValueError(
            "Unexpected maximum date."
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "location_id",
                "date",
            ]
        ).sum()
    )

    print(
        f"Duplicate city-date rows: "
        f"{duplicate_count}"
    )

    if duplicate_count != 0:
        raise ValueError(
            "Duplicate city-date records detected."
        )

    weather_columns = [
        "temperature_mean",
        "temperature_max",
        "temperature_min",
        "precipitation_sum",
        "relative_humidity_mean",
        "wind_speed_mean",
        "surface_pressure_mean",
    ]

    actual_missing_count = int(
        df[weather_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing weather values: "
        f"{actual_missing_count}"
    )

    if actual_missing_count != 0:
        raise ValueError(
            "Missing weather values detected."
        )

    invalid_temperature_count = int(
        (
            (df["temperature_min"] > df["temperature_mean"])
            | (
                df["temperature_mean"]
                > df["temperature_max"]
            )
        ).sum()
    )

    print(
        "Temperature consistency violations: "
        f"{invalid_temperature_count}"
    )

    if invalid_temperature_count != 0:
        raise ValueError(
            "Temperature min/mean/max consistency check failed."
        )

    rows_per_city = (
        df.groupby("city")
        .size()
    )

    if not (
        rows_per_city == 258
    ).all():
        raise ValueError(
            "Not every city has exactly 258 daily observations."
        )

    print(
        "Rows per city: 258"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    # -----------------------------------------------------------------
    # Write partitioned Parquet
    # -----------------------------------------------------------------

    print()
    print("-" * 80)
    print("WRITING PARQUET")
    print("-" * 80)

    # Remove old generated output so the final dataset is reproducible.
    for old_file in OUTPUT_DIR.rglob(
        "*.parquet"
    ):
        old_file.unlink()

    df["year"] = df["date"].dt.year

    for year, year_df in df.groupby(
        "year"
    ):

        year_dir = (
            OUTPUT_DIR
            / f"year={year}"
        )

        year_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            year_dir
            / "weather.parquet"
        )

        year_df.to_parquet(
            output_file,
            index=False,
            engine="pyarrow",
        )

        print(
            f"Year {year}: "
            f"{len(year_df):,} rows -> "
            f"{output_file}"
        )

    print()
    print("=" * 80)
    print(
        "RECENT IFS PARQUET CONVERSION COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
