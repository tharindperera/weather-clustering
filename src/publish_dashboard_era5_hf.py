from __future__ import annotations

from pathlib import Path

import pandas as pd
from huggingface_hub import (
    HfApi,
    get_token,
    hf_hub_download,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

HF_REPO_ID = (
    "tharinduperera/weather-clustering-data"
)

HF_FILE_PATH = (
    "processed/dashboard/"
    "weather_era5_2016_2025.parquet"
)

LOCAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dashboard"
    / "weather_era5_2016_2025.parquet"
)
if not LOCAL_FILE.exists() and (PROJECT_ROOT / "weather-clustering" / "data" / "processed" / "dashboard" / "weather_era5_2016_2025.parquet").exists():
    LOCAL_FILE = (
        PROJECT_ROOT
        / "weather-clustering"
        / "data"
        / "processed"
        / "dashboard"
        / "weather_era5_2016_2025.parquet"
    )


def validate(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():
        raise FileNotFoundError(
            f"ERA5 dashboard dataset not found: {path}"
        )

    df = pd.read_parquet(
        path
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    if len(df) != 365_300:
        raise ValueError(
            f"Expected 365,300 rows, found {len(df)}."
        )

    if df["location_id"].nunique() != 100:
        raise ValueError(
            "Expected exactly 100 locations."
        )

    if df.duplicated(
        ["location_id", "date"]
    ).any():
        raise ValueError(
            "Duplicate city-date rows found."
        )

    return df


def main() -> None:

    local_df = validate(
        LOCAL_FILE
    )

    token = get_token()

    if not token:
        raise RuntimeError(
            "No Hugging Face authentication token found."
        )

    api = HfApi(
        token=token
    )

    print(
        "Uploading consolidated ERA5 dashboard dataset..."
    )

    api.upload_file(
        path_or_fileobj=str(
            LOCAL_FILE
        ),
        path_in_repo=HF_FILE_PATH,
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        commit_message=(
            "Add consolidated ERA5 dashboard dataset"
        ),
    )

    print(
        "Upload complete."
    )

    # ---------------------------------------------------------------
    # Round-trip verification
    # ---------------------------------------------------------------

    downloaded = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILE_PATH,
        repo_type="dataset",
        force_download=True,
    )

    remote_df = validate(
        Path(downloaded)
    )

    print()
    print(
        "Round-trip verification successful."
    )

    print(
        f"Rows: {len(remote_df):,}"
    )

    print(
        f"Cities: {remote_df['city'].nunique()}"
    )

    print(
        f"Date range: "
        f"{remote_df['date'].min().date()} "
        f"to "
        f"{remote_df['date'].max().date()}"
    )


if __name__ == "__main__":
    main()
