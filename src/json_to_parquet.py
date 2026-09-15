from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from config import (
    CHECKPOINT_DIR,
    EXPECTED_LOCATION_COUNT,
    LOCATIONS_FILE,
    RAW_DIR,
)
from validator import validate_response


# =========================================================
# Configuration
# =========================================================

PARQUET_DIR = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "parquet"
    / "weather"
)

CONVERSION_MANIFEST = (
    CHECKPOINT_DIR
    / "parquet_manifest.json"
)

WORK_ID_PATTERN = re.compile(
    r"^(batch_(\d{4})__"
    r"(\d{4}-\d{2}-\d{2})__"
    r"(\d{4}-\d{2}-\d{2}))\.json$"
)


# =========================================================
# Utilities
# =========================================================

def load_locations() -> pd.DataFrame:
    if not LOCATIONS_FILE.exists():
        raise FileNotFoundError(
            f"Location catalogue not found: {LOCATIONS_FILE}"
        )

    df = pd.read_csv(LOCATIONS_FILE)

    if len(df) != EXPECTED_LOCATION_COUNT:
        raise ValueError(
            f"Expected exactly {EXPECTED_LOCATION_COUNT} "
            f"locations, found {len(df)}."
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
        "population",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Location catalogue is missing columns: {missing}"
        )

    return df.reset_index(drop=True)


def load_manifest() -> dict:
    if not CONVERSION_MANIFEST.exists():
        return {
            "version": 1,
            "items": {},
        }

    with CONVERSION_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_manifest(manifest: dict) -> None:
    CONVERSION_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = CONVERSION_MANIFEST.with_suffix(".tmp")

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )

    temporary.replace(
        CONVERSION_MANIFEST
    )


def parse_work_file(path: Path) -> tuple[str, int, str, str]:
    match = WORK_ID_PATTERN.match(path.name)

    if not match:
        raise ValueError(
            f"Unexpected raw filename: {path.name}"
        )

    return (
        match.group(1),
        int(match.group(2)),
        match.group(3),
        match.group(4),
    )


# =========================================================
# Normalization
# =========================================================

def normalize_raw_file(
    raw_file: Path,
    locations: pd.DataFrame,
) -> pd.DataFrame:

    work_id, batch_number, start_date, end_date = (
        parse_work_file(raw_file)
    )

    start_index = (
        (batch_number - 1)
        * 20
    )

    batch_locations = locations.iloc[
        start_index:start_index + 20
    ].reset_index(drop=True)

    if len(batch_locations) != 20:
        raise ValueError(
            f"{work_id}: expected 20 catalogue locations "
            f"for batch {batch_number}, found "
            f"{len(batch_locations)}."
        )

    with raw_file.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw_data = json.load(file)

    results = validate_response(
        data=raw_data,
        expected_locations=20,
        expected_days=(
            pd.Timestamp(end_date)
            - pd.Timestamp(start_date)
        ).days + 1,
        expected_start=start_date,
        expected_end=end_date,
    )

    rows: list[dict] = []

    for index, result in enumerate(results):

        catalogue = batch_locations.iloc[index]

        daily = result["daily"]

        dates = daily["time"]

        for day_index, day in enumerate(dates):

            rows.append(
                {
                    "location_id": int(
                        catalogue["location_id"]
                    ),
                    "city": catalogue["city"],
                    "city_ascii": catalogue["city_ascii"],
                    "country": catalogue["country"],
                    "iso2": catalogue["iso2"],
                    "iso3": catalogue["iso3"],
                    "admin_name": catalogue["admin_name"],
                    "capital": catalogue["capital"],
                    "source_latitude": float(
                        catalogue["latitude"]
                    ),
                    "source_longitude": float(
                        catalogue["longitude"]
                    ),
                    "model_latitude": float(
                        result["latitude"]
                    ),
                    "model_longitude": float(
                        result["longitude"]
                    ),
                    "timezone": result["timezone"],
                    "elevation": (
                        float(result["elevation"])
                        if result.get("elevation") is not None
                        else None
                    ),
                    "date": day,
                    "temperature_mean": daily[
                        "temperature_2m_mean"
                    ][day_index],
                    "temperature_max": daily[
                        "temperature_2m_max"
                    ][day_index],
                    "temperature_min": daily[
                        "temperature_2m_min"
                    ][day_index],
                    "precipitation_sum": daily[
                        "precipitation_sum"
                    ][day_index],
                    "relative_humidity_mean": daily[
                        "relative_humidity_2m_mean"
                    ][day_index],
                    "wind_speed_mean": daily[
                        "wind_speed_10m_mean"
                    ][day_index],
                    "surface_pressure_mean": daily[
                        "surface_pressure_mean"
                    ][day_index],
                }
            )

    df = pd.DataFrame(rows)

    expected_rows = (
        20
        * (
            (
                pd.Timestamp(end_date)
                - pd.Timestamp(start_date)
            ).days
            + 1
        )
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"{work_id}: expected {expected_rows} "
            f"normalized rows, found {len(df)}."
        )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="raise",
    )

    df["year"] = df["date"].dt.year.astype("int16")

    return df


# =========================================================
# Parquet writing
# =========================================================

def append_to_parquet(
    df: pd.DataFrame,
    work_id: str,
) -> None:

    PARQUET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pandas(
        df,
        preserve_index=False,
    )

    ds.write_dataset(
        table,
        base_dir=str(PARQUET_DIR),
        format="parquet",
        partitioning=ds.partitioning(
            pa.schema(
                [
                    pa.field(
                        "year",
                        pa.int16(),
                    )
                ]
            ),
            flavor="hive",
        ),
        basename_template=f"{work_id}_{{i}}.parquet",
        existing_data_behavior="overwrite_or_ignore",
    )


# =========================================================
# Main conversion
# =========================================================

def convert(
    max_files: int | None = None,
) -> None:

    locations = load_locations()
    manifest = load_manifest()

    raw_files = sorted(
        RAW_DIR.glob("*.json")
    )

    if not raw_files:
        print(
            "No raw historical JSON files found."
        )
        return

    print("=" * 70)
    print("JSON -> PARQUET CONVERSION")
    print("=" * 70)

    print(
        f"Raw JSON files found : {len(raw_files)}"
    )

    converted_now = 0

    for raw_file in raw_files:

        work_id, _, _, _ = parse_work_file(
            raw_file
        )

        # -------------------------------------------------
        # Skip already converted files
        # -------------------------------------------------

        if (
            work_id in manifest["items"]
            and manifest["items"][work_id]["status"]
            == "success"
        ):
            print(
                f"SKIP {work_id} "
                "(already converted)"
            )
            continue

        # -------------------------------------------------
        # Convert
        # -------------------------------------------------

        print(
            f"\nProcessing: {work_id}"
        )

        try:

            df = normalize_raw_file(
                raw_file,
                locations,
            )

            append_to_parquet(
                df,
                work_id,
            )

            manifest["items"][work_id] = {
                "status": "success",
                "raw_file": str(raw_file),
                "rows": int(len(df)),
            }

            save_manifest(
                manifest
            )

            converted_now += 1

            print(
                f"SUCCESS: {work_id} "
                f"({len(df)} rows)"
            )

        except Exception as exc:

            manifest["items"][work_id] = {
                "status": "failed",
                "raw_file": str(raw_file),
                "error": str(exc),
            }

            save_manifest(
                manifest
            )

            print(
                f"FAILED: {work_id}"
            )

            print(
                f"Error: {exc}"
            )

            raise

        if (
            max_files is not None
            and converted_now >= max_files
        ):
            print(
                f"\nReached max files: "
                f"{max_files}"
            )
            return

    print(
        "\nConversion completed."
    )


# =========================================================
# CLI
# =========================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Incrementally convert validated "
            "Open-Meteo JSON data to Parquet."
        )
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "Convert at most this many raw JSON "
            "files during this run."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Show raw-file counts without "
            "converting anything."
        ),
    )

    args = parser.parse_args()

    if args.dry_run:

        locations = load_locations()
        raw_files = sorted(
            RAW_DIR.glob("*.json")
        )
        manifest = load_manifest()

        converted = sum(
            1
            for item in manifest["items"].values()
            if item["status"] == "success"
        )

        print("=" * 70)
        print("JSON -> PARQUET DRY RUN")
        print("=" * 70)

        print(
            f"Locations           : {len(locations)}"
        )

        print(
            f"Raw JSON files      : {len(raw_files)}"
        )

        print(
            f"Already converted   : {converted}"
        )

        print(
            f"Pending conversion  : "
            f"{len(raw_files) - converted}"
        )

        print(
            f"Parquet directory   : {PARQUET_DIR}"
        )

        return

    convert(
        max_files=args.max_files
    )


if __name__ == "__main__":
    main()