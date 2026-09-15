from __future__ import annotations

import sys
from pathlib import Path
import duckdb
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)

PARQUET_PATH = "data/processed/parquet/weather/**/*.parquet"


def main() -> None:
    con = duckdb.connect()

    print("=" * 80)
    print("PHASE 3: DUCKDB EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. DATASET SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. DATASET SUMMARY")
    print("=" * 80)
    summary_df = con.execute(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT location_id) AS cities,
            COUNT(DISTINCT country) AS countries,
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            COUNT(DISTINCT year) AS total_years
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        """
    ).fetchdf()
    print(summary_df.to_string(index=False))

    # -------------------------------------------------------------
    # 2. ROW COUNT BY CITY
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. ROW COUNT BY CITY (Sample 15 & Distribution Check)")
    print("=" * 80)
    city_rows_df = con.execute(
        f"""
        SELECT
            city,
            country,
            COUNT(*) AS row_count,
            MIN(date) AS min_date,
            MAX(date) AS max_date
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        GROUP BY city, country
        ORDER BY city
        """
    ).fetchdf()
    
    print(f"Total cities evaluated: {len(city_rows_df)}")
    print(f"Unique row counts across cities: {city_rows_df['row_count'].unique()}")
    print("\nFirst 15 cities sample:")
    print(city_rows_df.head(15).to_string(index=False))

    # -------------------------------------------------------------
    # 3. MISSING VALUES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. MISSING VALUES")
    print("=" * 80)
    null_counts_df = con.execute(
        f"""
        SELECT
            SUM(CASE WHEN location_id IS NULL THEN 1 ELSE 0 END) AS null_location_id,
            SUM(CASE WHEN city IS NULL THEN 1 ELSE 0 END) AS null_city,
            SUM(CASE WHEN country IS NULL THEN 1 ELSE 0 END) AS null_country,
            SUM(CASE WHEN date IS NULL THEN 1 ELSE 0 END) AS null_date,
            SUM(CASE WHEN temperature_mean IS NULL THEN 1 ELSE 0 END) AS null_temp_mean,
            SUM(CASE WHEN temperature_max IS NULL THEN 1 ELSE 0 END) AS null_temp_max,
            SUM(CASE WHEN temperature_min IS NULL THEN 1 ELSE 0 END) AS null_temp_min,
            SUM(CASE WHEN precipitation_sum IS NULL THEN 1 ELSE 0 END) AS null_precip_sum,
            SUM(CASE WHEN relative_humidity_mean IS NULL THEN 1 ELSE 0 END) AS null_humidity,
            SUM(CASE WHEN wind_speed_mean IS NULL THEN 1 ELSE 0 END) AS null_wind_speed,
            SUM(CASE WHEN surface_pressure_mean IS NULL THEN 1 ELSE 0 END) AS null_pressure
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        """
    ).fetchdf()
    print(null_counts_df.to_string(index=False))

    # -------------------------------------------------------------
    # 4. GLOBAL WEATHER STATISTICS
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. GLOBAL WEATHER STATISTICS")
    print("=" * 80)
    global_stats_df = con.execute(
        f"""
        SELECT
            'temperature_mean (°C)' AS variable,
            ROUND(AVG(temperature_mean), 2) AS mean,
            ROUND(STDDEV(temperature_mean), 2) AS stddev,
            ROUND(MIN(temperature_mean), 2) AS min,
            ROUND(QUANTILE_CONT(temperature_mean, 0.25), 2) AS q25,
            ROUND(MEDIAN(temperature_mean), 2) AS median,
            ROUND(QUANTILE_CONT(temperature_mean, 0.75), 2) AS q75,
            ROUND(MAX(temperature_mean), 2) AS max
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        UNION ALL
        SELECT
            'temperature_max (°C)',
            ROUND(AVG(temperature_max), 2),
            ROUND(STDDEV(temperature_max), 2),
            ROUND(MIN(temperature_max), 2),
            ROUND(QUANTILE_CONT(temperature_max, 0.25), 2),
            ROUND(MEDIAN(temperature_max), 2),
            ROUND(QUANTILE_CONT(temperature_max, 0.75), 2),
            ROUND(MAX(temperature_max), 2)
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        UNION ALL
        SELECT
            'temperature_min (°C)',
            ROUND(AVG(temperature_min), 2),
            ROUND(STDDEV(temperature_min), 2),
            ROUND(MIN(temperature_min), 2),
            ROUND(QUANTILE_CONT(temperature_min, 0.25), 2),
            ROUND(MEDIAN(temperature_min), 2),
            ROUND(QUANTILE_CONT(temperature_min, 0.75), 2),
            ROUND(MAX(temperature_min), 2)
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        UNION ALL
        SELECT
            'precipitation_sum (mm)',
            ROUND(AVG(precipitation_sum), 2),
            ROUND(STDDEV(precipitation_sum), 2),
            ROUND(MIN(precipitation_sum), 2),
            ROUND(QUANTILE_CONT(precipitation_sum, 0.25), 2),
            ROUND(MEDIAN(precipitation_sum), 2),
            ROUND(QUANTILE_CONT(precipitation_sum, 0.75), 2),
            ROUND(MAX(precipitation_sum), 2)
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        UNION ALL
        SELECT
            'relative_humidity_mean (%)',
            ROUND(AVG(relative_humidity_mean), 2),
            ROUND(STDDEV(relative_humidity_mean), 2),
            ROUND(MIN(relative_humidity_mean), 2),
            ROUND(QUANTILE_CONT(relative_humidity_mean, 0.25), 2),
            ROUND(MEDIAN(relative_humidity_mean), 2),
            ROUND(QUANTILE_CONT(relative_humidity_mean, 0.75), 2),
            ROUND(MAX(relative_humidity_mean), 2)
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        UNION ALL
        SELECT
            'wind_speed_mean (km/h)',
            ROUND(AVG(wind_speed_mean), 2),
            ROUND(STDDEV(wind_speed_mean), 2),
            ROUND(MIN(wind_speed_mean), 2),
            ROUND(QUANTILE_CONT(wind_speed_mean, 0.25), 2),
            ROUND(MEDIAN(wind_speed_mean), 2),
            ROUND(QUANTILE_CONT(wind_speed_mean, 0.75), 2),
            ROUND(MAX(wind_speed_mean), 2)
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        UNION ALL
        SELECT
            'surface_pressure_mean (hPa)',
            ROUND(AVG(surface_pressure_mean), 2),
            ROUND(STDDEV(surface_pressure_mean), 2),
            ROUND(MIN(surface_pressure_mean), 2),
            ROUND(QUANTILE_CONT(surface_pressure_mean, 0.25), 2),
            ROUND(MEDIAN(surface_pressure_mean), 2),
            ROUND(QUANTILE_CONT(surface_pressure_mean, 0.75), 2),
            ROUND(MAX(surface_pressure_mean), 2)
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        """
    ).fetchdf()
    print(global_stats_df.to_string(index=False))

    # -------------------------------------------------------------
    # 5. COLDEST 10 CITIES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. COLDEST 10 CITIES (By Mean Temperature)")
    print("=" * 80)
    coldest_df = con.execute(
        f"""
        SELECT
            city,
            country,
            ROUND(AVG(temperature_mean), 2) AS avg_temp_mean,
            ROUND(MIN(temperature_min), 2) AS min_temp_recorded,
            ROUND(MAX(temperature_max), 2) AS max_temp_recorded
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        GROUP BY city, country
        ORDER BY avg_temp_mean ASC
        LIMIT 10
        """
    ).fetchdf()
    print(coldest_df.to_string(index=False))

    # -------------------------------------------------------------
    # 6. HOTTEST 10 CITIES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. HOTTEST 10 CITIES (By Mean Temperature)")
    print("=" * 80)
    hottest_df = con.execute(
        f"""
        SELECT
            city,
            country,
            ROUND(AVG(temperature_mean), 2) AS avg_temp_mean,
            ROUND(MIN(temperature_min), 2) AS min_temp_recorded,
            ROUND(MAX(temperature_max), 2) AS max_temp_recorded
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        GROUP BY city, country
        ORDER BY avg_temp_mean DESC
        LIMIT 10
        """
    ).fetchdf()
    print(hottest_df.to_string(index=False))

    # -------------------------------------------------------------
    # 7. WETTEST 10 CITIES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("7. WETTEST 10 CITIES (By Total & Mean Annual Precipitation)")
    print("=" * 80)
    wettest_df = con.execute(
        f"""
        SELECT
            city,
            country,
            ROUND(SUM(precipitation_sum) / 10.0, 2) AS avg_annual_precip_mm,
            ROUND(AVG(precipitation_sum), 2) AS avg_daily_precip_mm,
            ROUND(MAX(precipitation_sum), 2) AS max_single_day_precip_mm
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        GROUP BY city, country
        ORDER BY avg_annual_precip_mm DESC
        LIMIT 10
        """
    ).fetchdf()
    print(wettest_df.to_string(index=False))

    # -------------------------------------------------------------
    # 8. DRIEST 10 CITIES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("8. DRIEST 10 CITIES (By Total & Mean Annual Precipitation)")
    print("=" * 80)
    driest_df = con.execute(
        f"""
        SELECT
            city,
            country,
            ROUND(SUM(precipitation_sum) / 10.0, 2) AS avg_annual_precip_mm,
            ROUND(AVG(precipitation_sum), 2) AS avg_daily_precip_mm,
            ROUND(MAX(precipitation_sum), 2) AS max_single_day_precip_mm
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        GROUP BY city, country
        ORDER BY avg_annual_precip_mm ASC
        LIMIT 10
        """
    ).fetchdf()
    print(driest_df.to_string(index=False))

    # -------------------------------------------------------------
    # 9. GLOBAL MONTHLY TEMPERATURE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("9. GLOBAL MONTHLY TEMPERATURE PATTERNS")
    print("=" * 80)
    monthly_temp_df = con.execute(
        f"""
        SELECT
            EXTRACT(MONTH FROM date) AS month,
            ROUND(AVG(temperature_mean), 2) AS global_avg_temp_mean,
            ROUND(AVG(temperature_max), 2) AS global_avg_temp_max,
            ROUND(AVG(temperature_min), 2) AS global_avg_temp_min
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        GROUP BY month
        ORDER BY month
        """
    ).fetchdf()
    print(monthly_temp_df.to_string(index=False))

    # -------------------------------------------------------------
    # 10. GLOBAL MONTHLY PRECIPITATION
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("10. GLOBAL MONTHLY PRECIPITATION PATTERNS")
    print("=" * 80)
    monthly_precip_df = con.execute(
        f"""
        SELECT
            EXTRACT(MONTH FROM date) AS month,
            ROUND(AVG(precipitation_sum), 2) AS avg_daily_precip_mm,
            ROUND(SUM(precipitation_sum) / 10.0 / 100.0, 2) AS avg_monthly_precip_per_city_mm
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        GROUP BY month
        ORDER BY month
        """
    ).fetchdf()
    print(monthly_precip_df.to_string(index=False))

    # -------------------------------------------------------------
    # 11. DUPLICATE CITY-DATE CHECK
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("11. DUPLICATE CITY-DATE CHECK")
    print("=" * 80)
    dups_df = con.execute(
        f"""
        SELECT
            location_id,
            city,
            date,
            COUNT(*) AS occurrence_count
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        GROUP BY location_id, city, date
        HAVING COUNT(*) > 1
        """
    ).fetchdf()
    print(f"Duplicates found: {len(dups_df)}")

    # -------------------------------------------------------------
    # 12. TEMPERATURE CONSISTENCY CHECK
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("12. TEMPERATURE CONSISTENCY CHECK (min <= mean <= max)")
    print("=" * 80)
    temp_consistency_df = con.execute(
        f"""
        SELECT
            SUM(CASE WHEN temperature_min > temperature_mean THEN 1 ELSE 0 END) AS min_greater_than_mean,
            SUM(CASE WHEN temperature_mean > temperature_max THEN 1 ELSE 0 END) AS mean_greater_than_max,
            SUM(CASE WHEN temperature_min > temperature_max THEN 1 ELSE 0 END) AS min_greater_than_max
        FROM read_parquet('{PARQUET_PATH}', hive_partitioning=true)
        """
    ).fetchdf()
    print(temp_consistency_df.to_string(index=False))

    con.close()
    print("\n" + "=" * 80)
    print("DUCKDB EDA COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
