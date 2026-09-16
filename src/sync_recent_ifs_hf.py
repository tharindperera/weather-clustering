from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from huggingface_hub import HfApi, get_token as hf_get_token, hf_hub_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]

HF_REPO_ID = "tharinduperera/weather-clustering-data"
HF_PARQUET_PATH = (
    "processed/parquet/"
    "weather_ifs_2026-present/"
    "year=2026/"
    "weather.parquet"
)

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


def get_token() -> str:
    token = os.environ.get("HF_TOKEN") or hf_get_token()
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set and no cached Hugging Face token was found."
        )
    return token


def validate_local_file() -> pd.DataFrame:
    if not LOCAL_PARQUET.exists():
        raise FileNotFoundError(
            f"Recent IFS Parquet not found: {LOCAL_PARQUET}"
        )

    df = pd.read_parquet(LOCAL_PARQUET)

    required = {
        "location_id",
        "city",
        "date",
        "temperature_mean",
        "precipitation_sum",
        "relative_humidity_mean",
        "wind_speed_mean",
        "surface_pressure_mean",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if df["city"].nunique() != 100:
        raise ValueError(
            "Expected exactly 100 unique cities."
        )

    df["date"] = pd.to_datetime(df["date"])
    return df


def upload() -> None:
    token = get_token()
    df = validate_local_file()

    latest_date = (
        df["date"]
        .max()
        .date()
    )

    api = HfApi(token=token)

    api.upload_file(
        path_or_fileobj=str(LOCAL_PARQUET),
        path_in_repo=HF_PARQUET_PATH,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=(
            "Update ECMWF IFS recent weather "
            f"through {latest_date}"
        ),
    )

    print("Uploaded recent IFS dataset.")
    print(f"Rows: {len(df):,}")
    print(f"Cities: {df['city'].nunique()}")
    print(f"Latest date: {latest_date}")


def download() -> None:
    token = get_token()

    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_PARQUET_PATH,
        repo_type="dataset",
        token=token,
    )

    LOCAL_PARQUET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        downloaded,
        LOCAL_PARQUET,
    )

    df = validate_local_file()

    print("Downloaded recent IFS dataset.")
    print(f"Rows: {len(df):,}")
    print(
        f"Latest date: "
        f"{df['date'].max().date()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=[
            "upload",
            "download",
        ],
    )
    args = parser.parse_args()

    if args.action == "upload":
        upload()
    else:
        download()


if __name__ == "__main__":
    main()
