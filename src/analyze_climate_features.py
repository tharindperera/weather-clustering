from __future__ import annotations

import sys
from pathlib import Path
import duckdb
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "climate_features.parquet"
)

FEATURES = [
    "mean_temperature",
    "temperature_std",
    "mean_diurnal_range",
    "temperature_p10",
    "temperature_p90",
    "temperature_p90_p10_range",
    "temperature_seasonality",
    "annual_precipitation",
    "precipitation_std",
    "wet_day_frequency",
    "maximum_daily_precipitation",
    "precipitation_seasonality",
    "mean_humidity",
    "humidity_std",
    "mean_wind",
    "wind_std",
]


def main() -> None:

    con = duckdb.connect()

    print("=" * 80)
    print("CLIMATE FEATURE DIAGNOSTICS")
    print("=" * 80)

    df = con.execute(
        f"""
        SELECT *
        FROM read_parquet(
            '{FEATURE_FILE}'
        )
        """
    ).fetchdf()

    print("\nDataset shape:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # -----------------------------------------------------
    # 1. Feature statistics
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("1. FEATURE STATISTICS")
    print("=" * 80)

    stats = df[FEATURES].describe().T

    stats["range"] = (
        stats["max"]
        - stats["min"]
    )

    print(
        stats[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
                "range",
            ]
        ].round(3).to_string()
    )

    # -----------------------------------------------------
    # 2. Skewness
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("2. SKEWNESS")
    print("=" * 80)

    skewness = (
        df[FEATURES]
        .skew()
        .sort_values(
            ascending=False
        )
    )

    print(
        skewness.round(3).to_string()
    )

    # -----------------------------------------------------
    # 3. Correlation matrix
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("3. ABSOLUTE CORRELATIONS (Top 20)")
    print("=" * 80)

    corr = (
        df[FEATURES]
        .corr()
        .abs()
    )

    pairs = []

    for i in range(
        len(FEATURES)
    ):

        for j in range(
            i + 1,
            len(FEATURES),
        ):

            pairs.append(
                (
                    FEATURES[i],
                    FEATURES[j],
                    corr.iloc[i, j],
                )
            )

    pairs_df = pd.DataFrame(
        pairs,
        columns=[
            "feature_1",
            "feature_2",
            "absolute_correlation",
        ],
    ).sort_values(
        "absolute_correlation",
        ascending=False,
    )

    print(
        pairs_df.head(20)
        .round(3)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # 4. Feature ranges
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("4. FEATURE RANGES")
    print("=" * 80)

    ranges = (
        df[FEATURES]
        .agg(
            ["min", "max"]
        )
        .T
    )

    ranges["range"] = (
        ranges["max"]
        - ranges["min"]
    )

    print(
        ranges
        .sort_values(
            "range",
            ascending=False,
        )
        .round(3)
        .to_string()
    )

    # -----------------------------------------------------
    # 5. Extremal cities
    # -----------------------------------------------------

    print("\n" + "=" * 80)
    print("5. EXTREME CLIMATE FEATURE VALUES")
    print("=" * 80)

    for feature in [
        "mean_temperature",
        "annual_precipitation",
        "wet_day_frequency",
        "mean_humidity",
        "mean_wind",
        "temperature_seasonality",
        "precipitation_seasonality",
    ]:

        print(
            f"\nHighest {feature}:"
        )

        print(
            df[
                [
                    "city",
                    "country",
                    feature,
                ]
            ]
            .sort_values(
                feature,
                ascending=False,
            )
            .head(5)
            .round(3)
            .to_string(index=False)
        )

        print(
            f"\nLowest {feature}:"
        )

        print(
            df[
                [
                    "city",
                    "country",
                    feature,
                ]
            ]
            .sort_values(
                feature,
                ascending=True,
            )
            .head(5)
            .round(3)
            .to_string(index=False)
        )

    con.close()

    print(
        "\nFeature diagnostics completed."
    )


if __name__ == "__main__":
    main()
