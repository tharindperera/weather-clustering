from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PROCESSED_DIR

INPUT_FILE = PROCESSED_DIR / "features" / "clustering_input_original5.parquet"

WEATHER_COLUMNS = [
    "temperature_mean",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]


def main() -> None:
    print("=" * 80)
    print("CLUSTERING INPUT DIAGNOSTICS")
    print("=" * 80)

    df = pd.read_parquet(INPUT_FILE)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # ------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------

    if len(df) != 100:
        raise ValueError(f"Expected 100 cities, found {len(df)}.")

    if df["city"].nunique() != 100:
        raise ValueError("Expected 100 unique cities.")

    if df[WEATHER_COLUMNS].isna().any().any():
        raise ValueError("Missing values detected.")

    if not np.isfinite(df[WEATHER_COLUMNS].to_numpy()).all():
        raise ValueError("Infinite/non-finite values detected.")

    # ------------------------------------------------------------------
    # 1. Descriptive statistics
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("1. DESCRIPTIVE STATISTICS")
    print("=" * 80)

    stats = df[WEATHER_COLUMNS].describe().T
    stats["range"] = stats["max"] - stats["min"]

    print(stats.round(3).to_string())

    # ------------------------------------------------------------------
    # 2. Skewness
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("2. SKEWNESS")
    print("=" * 80)

    skewness = df[WEATHER_COLUMNS].skew().sort_values(
        ascending=False
    )

    print(skewness.round(3).to_string())

    # ------------------------------------------------------------------
    # 3. Correlations
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("3. CORRELATION MATRIX")
    print("=" * 80)

    correlation = df[WEATHER_COLUMNS].corr()

    print(correlation.round(3).to_string())

    # Absolute correlations excluding diagonal.
    pairs = []

    for i, feature_a in enumerate(WEATHER_COLUMNS):
        for j, feature_b in enumerate(WEATHER_COLUMNS):
            if j <= i:
                continue

            corr = correlation.loc[feature_a, feature_b]

            pairs.append(
                {
                    "feature_1": feature_a,
                    "feature_2": feature_b,
                    "correlation": corr,
                    "absolute_correlation": abs(corr),
                }
            )

    pair_df = (
        pd.DataFrame(pairs)
        .sort_values("absolute_correlation", ascending=False)
        .reset_index(drop=True)
    )

    print()
    print("Correlation pairs:")
    print(pair_df.round(3).to_string(index=False))

    # ------------------------------------------------------------------
    # 4. Extreme cities
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("4. EXTREME CITY VALUES")
    print("=" * 80)

    for feature in WEATHER_COLUMNS:
        print()
        print(f"Highest {feature}:")
        print(
            df.nlargest(5, feature)[
                ["city", "country", feature]
            ].to_string(index=False)
        )

        print()
        print(f"Lowest {feature}:")
        print(
            df.nsmallest(5, feature)[
                ["city", "country", feature]
            ].to_string(index=False)
        )

    # ------------------------------------------------------------------
    # 5. Final validation
    # ------------------------------------------------------------------

    print()
    print("=" * 80)
    print("5. VALIDATION")
    print("=" * 80)

    print(f"Unique cities: {df['city'].nunique()}")
    print(
        f"Missing weather values: "
        f"{int(df[WEATHER_COLUMNS].isna().sum().sum())}"
    )
    print(
        f"Non-finite weather values: "
        f"{int((~np.isfinite(df[WEATHER_COLUMNS])).sum().sum())}"
    )

    print()
    print("=" * 80)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
