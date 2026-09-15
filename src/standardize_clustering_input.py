from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PROCESSED_DIR

INPUT_FILE = PROCESSED_DIR / "features" / "clustering_input_original5.parquet"
OUTPUT_FILE = PROCESSED_DIR / "features" / "clustering_standardized5.parquet"

WEATHER_COLUMNS = [
    "temperature_mean",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]


def main() -> None:
    print("=" * 80)
    print("STANDARDIZE CLUSTERING INPUT")
    print("=" * 80)

    df = pd.read_parquet(INPUT_FILE)

    if len(df) != 100:
        raise ValueError(
            f"Expected 100 cities, found {len(df)}."
        )

    if df["city"].nunique() != 100:
        raise ValueError(
            "Expected 100 unique cities."
        )

    if df[WEATHER_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing values detected."
        )

    values = df[WEATHER_COLUMNS].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "Non-finite values detected."
        )

    # ---------------------------------------------------------------
    # Standardization
    # ---------------------------------------------------------------

    scaler = StandardScaler()

    standardized_values = scaler.fit_transform(values)

    standardized_columns = [
        f"{column}_z"
        for column in WEATHER_COLUMNS
    ]

    standardized_df = pd.DataFrame(
        standardized_values,
        columns=standardized_columns,
        index=df.index,
    )

    output_df = pd.concat(
        [
            df[
                [
                    "location_id",
                    "city",
                    "city_ascii",
                    "country",
                    "iso2",
                    "iso3",
                ]
            ].reset_index(drop=True),
            standardized_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # ---------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------

    print()
    print("Standardized feature statistics:")

    validation = output_df[standardized_columns].agg(
        ["mean", "std", "min", "max"]
    )

    print(validation.round(6).to_string())

    means = output_df[standardized_columns].mean().abs()

    if not (means < 1e-10).all():
        raise ValueError(
            "Standardized means are not approximately zero."
        )

    if output_df[standardized_columns].isna().any().any():
        raise ValueError(
            "Missing values detected after standardization."
        )

    if not np.isfinite(
        output_df[standardized_columns].to_numpy()
    ).all():
        raise ValueError(
            "Non-finite values detected after standardization."
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
    )

    print()
    print(f"Rows: {len(output_df):,}")
    print(f"Columns: {len(output_df.columns):,}")

    print()
    print("Output:")
    print(f"  {OUTPUT_FILE}")

    print()
    print("=" * 80)
    print("STANDARDIZATION VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
