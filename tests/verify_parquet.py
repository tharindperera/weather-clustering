import duckdb

PARQUET_PATH = (
    "data/processed/parquet/weather/**/*.parquet"
)

con = duckdb.connect()

print("=" * 70)
print("PARQUET / DUCKDB VERIFICATION")
print("=" * 70)

row_count = con.execute(
    f"""
    SELECT COUNT(*)
    FROM read_parquet(
        '{PARQUET_PATH}',
        hive_partitioning=true
    )
    """
).fetchone()[0]

print(f"\nRows: {row_count:,}")

print("\nSchema:")
print(
    con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet(
            '{PARQUET_PATH}',
            hive_partitioning=true
        )
        """
    ).fetchdf().to_string(index=False)
)

print("\nCoverage:")
print(
    con.execute(
        f"""
        SELECT
            MIN(date) AS min_date,
            MAX(date) AS max_date,
            COUNT(DISTINCT location_id) AS cities
        FROM read_parquet(
            '{PARQUET_PATH}',
            hive_partitioning=true
        )
        """
    ).fetchdf().to_string(index=False)
)

print("\nSample rows:")
print(
    con.execute(
        f"""
        SELECT
            location_id,
            city,
            country,
            source_latitude,
            source_longitude,
            model_latitude,
            model_longitude,
            date,
            temperature_mean,
            precipitation_sum
        FROM read_parquet(
            '{PARQUET_PATH}',
            hive_partitioning=true
        )
        ORDER BY location_id, date
        LIMIT 10
        """
    ).fetchdf().to_string(index=False)
)

con.close()
