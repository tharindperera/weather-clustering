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

PARQUET_GLOB = (
    str(
        PROJECT_ROOT
        / "data"
        / "processed"
        / "parquet"
        / "weather"
        / "**"
        / "*.parquet"
    )
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "climate_features.parquet"
)


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()

    print("=" * 80)
    print("BUILDING CITY-LEVEL CLIMATE FEATURES")
    print("=" * 80)

    query = f"""
    WITH daily AS (

        SELECT
            location_id,
            city,
            city_ascii,
            country,
            iso2,
            iso3,
            date,

            temperature_mean,
            temperature_max,
            temperature_min,

            precipitation_sum,
            relative_humidity_mean,
            wind_speed_mean

        FROM read_parquet(
            '{PARQUET_GLOB}',
            hive_partitioning = true
        )
    ),

    base_features AS (

        SELECT

            location_id,
            city,
            city_ascii,
            country,
            iso2,
            iso3,

            COUNT(*) AS observation_days,

            AVG(temperature_mean)
                AS mean_temperature,

            STDDEV_SAMP(temperature_mean)
                AS temperature_std,

            AVG(
                temperature_max
                - temperature_min
            )
                AS mean_diurnal_range,

            QUANTILE_CONT(
                temperature_mean,
                0.10
            )
                AS temperature_p10,

            QUANTILE_CONT(
                temperature_mean,
                0.90
            )
                AS temperature_p90,

            SUM(precipitation_sum)
                / 10.0
                AS annual_precipitation,

            STDDEV_SAMP(
                precipitation_sum
            )
                AS precipitation_std,

            AVG(
                CASE
                    WHEN precipitation_sum > 1.0
                    THEN 1.0
                    ELSE 0.0
                END
            )
                AS wet_day_frequency,

            MAX(precipitation_sum)
                AS maximum_daily_precipitation,

            AVG(relative_humidity_mean)
                AS mean_humidity,

            STDDEV_SAMP(
                relative_humidity_mean
            )
                AS humidity_std,

            AVG(wind_speed_mean)
                AS mean_wind,

            STDDEV_SAMP(
                wind_speed_mean
            )
                AS wind_std

        FROM daily

        GROUP BY
            location_id,
            city,
            city_ascii,
            country,
            iso2,
            iso3
    ),

    monthly_temperature AS (

        SELECT

            location_id,

            EXTRACT(
                MONTH FROM date
            ) AS month,

            AVG(
                temperature_mean
            ) AS monthly_temperature

        FROM daily

        GROUP BY
            location_id,
            month
    ),

    temperature_seasonality AS (

        SELECT

            location_id,

            MAX(
                monthly_temperature
            )
            -
            MIN(
                monthly_temperature
            )
                AS temperature_seasonality

        FROM monthly_temperature

        GROUP BY location_id
    ),

    monthly_precipitation AS (

        SELECT

            location_id,

            EXTRACT(
                MONTH FROM date
            ) AS month,

            SUM(
                precipitation_sum
            )
            / 10.0
                AS monthly_precipitation

        FROM daily

        GROUP BY
            location_id,
            month
    ),

    precipitation_seasonality AS (

        SELECT

            location_id,

            MAX(
                monthly_precipitation
            )
            -
            MIN(
                monthly_precipitation
            )
                AS precipitation_seasonality

        FROM monthly_precipitation

        GROUP BY location_id
    )

    SELECT

        b.location_id,
        b.city,
        b.city_ascii,
        b.country,
        b.iso2,
        b.iso3,

        b.observation_days,

        b.mean_temperature,
        b.temperature_std,
        b.mean_diurnal_range,

        b.temperature_p10,
        b.temperature_p90,

        b.temperature_p90
            - b.temperature_p10
            AS temperature_p90_p10_range,

        t.temperature_seasonality,

        b.annual_precipitation,
        b.precipitation_std,
        b.wet_day_frequency,
        b.maximum_daily_precipitation,

        p.precipitation_seasonality,

        b.mean_humidity,
        b.humidity_std,

        b.mean_wind,
        b.wind_std

    FROM base_features b

    INNER JOIN temperature_seasonality t
        ON b.location_id = t.location_id

    INNER JOIN precipitation_seasonality p
        ON b.location_id = p.location_id

    ORDER BY b.city
    """

    features = con.execute(
        query
    ).fetchdf()

    print(
        f"\nCities generated: "
        f"{len(features)}"
    )

    print(
        f"Features generated: "
        f"{len(features.columns)}"
    )

    print("\nFeature table:")
    print(
        features.to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # Quality checks
    # -------------------------------------------------

    print("\n" + "=" * 80)
    print("FEATURE QUALITY CHECKS")
    print("=" * 80)

    if len(features) != 100:
        raise RuntimeError(
            f"Expected 100 cities, "
            f"found {len(features)}."
        )

    duplicate_count = (
        features["location_id"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate cities:",
        duplicate_count,
    )

    if duplicate_count != 0:
        raise RuntimeError(
            "Duplicate location_id values detected."
        )

    numeric_columns = [
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

    missing = (
        features[numeric_columns]
        .isna()
        .sum()
    )

    print("\nMissing feature values:")
    print(missing.to_string())

    if missing.sum() != 0:
        raise RuntimeError(
            "Missing values detected in "
            "the clustering feature table."
        )

    # -------------------------------------------------
    # Save Parquet
    # -------------------------------------------------

    features.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved feature table to:\n"
        f"{OUTPUT_FILE}"
    )

    con.close()

    print(
        "\nClimate feature engineering "
        "completed successfully."
    )


if __name__ == "__main__":
    main()
