from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from huggingface_hub import hf_hub_download


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOCATIONS_FILE = (
    PROJECT_ROOT
    / "data"
    / "locations"
    / "locations.csv"
)
if not LOCATIONS_FILE.exists() and (PROJECT_ROOT / "weather-clustering" / "data" / "locations" / "locations.csv").exists():
    LOCATIONS_FILE = (
        PROJECT_ROOT
        / "weather-clustering"
        / "data"
        / "locations"
        / "locations.csv"
    )

CLUSTER_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clustering"
    / "dashboard_clustering_dataset.parquet"
)
if not CLUSTER_FILE.exists() and (PROJECT_ROOT / "weather-clustering" / "data" / "processed" / "clustering" / "dashboard_clustering_dataset.parquet").exists():
    CLUSTER_FILE = (
        PROJECT_ROOT
        / "weather-clustering"
        / "data"
        / "processed"
        / "clustering"
        / "dashboard_clustering_dataset.parquet"
    )

CLIMATE_FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "features"
    / "climate_features.parquet"
)
if not CLIMATE_FEATURE_FILE.exists() and (PROJECT_ROOT / "weather-clustering" / "data" / "processed" / "features" / "climate_features.parquet").exists():
    CLIMATE_FEATURE_FILE = (
        PROJECT_ROOT
        / "weather-clustering"
        / "data"
        / "processed"
        / "features"
        / "climate_features.parquet"
    )

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "qa"
)

REPORT_FILE = (
    REPORT_DIR
    / "phase_11c_data_quality_provenance.json"
)


# =============================================================================
# HUGGING FACE
# =============================================================================

HF_REPO_ID = "tharinduperera/weather-clustering-data"

HF_ERA5_FILE = (
    "processed/dashboard/"
    "weather_era5_2016_2025.parquet"
)

HF_IFS_FILE = (
    "processed/parquet/"
    "weather_ifs_2026-present/"
    "year=2026/"
    "weather.parquet"
)


# =============================================================================
# EXPECTED WEATHER VARIABLES
# =============================================================================

WEATHER_COLUMNS = [
    "temperature_mean",
    "temperature_max",
    "temperature_min",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]

CLUSTERING_COLUMNS = [
    "temperature_c",
    "rainfall_mm_per_day",
    "humidity_percent",
    "wind_speed_kmh",
    "pressure_hpa",
]

CLIMATE_FEATURE_COLUMNS = [
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


# =============================================================================
# HELPERS
# =============================================================================

def assert_equal(
    actual,
    expected,
    message: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            f"{message}: expected {expected!r}, got {actual!r}"
        )


def validate_daily_weather(
    df: pd.DataFrame,
    *,
    name: str,
    expected_start: str,
    expected_end: str | None,
    expected_locations: int = 100,
) -> dict:

    df = df.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    missing_columns = [
        column
        for column in WEATHER_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise AssertionError(
            f"{name}: missing weather columns: {missing_columns}"
        )

    location_count = int(
        df["location_id"].nunique()
    )

    city_count = int(
        df["city"].nunique()
    )

    min_date = (
        df["date"]
        .min()
        .date()
    )

    max_date = (
        df["date"]
        .max()
        .date()
    )

    assert_equal(
        location_count,
        expected_locations,
        f"{name}: location count",
    )

    assert_equal(
        city_count,
        expected_locations,
        f"{name}: city count",
    )

    assert_equal(
        str(min_date),
        expected_start,
        f"{name}: minimum date",
    )

    if expected_end is not None:
        assert_equal(
            str(max_date),
            expected_end,
            f"{name}: maximum date",
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "location_id",
                "date",
            ]
        ).sum()
    )

    missing_weather_values = int(
        df[
            WEATHER_COLUMNS
        ]
        .isna()
        .sum()
        .sum()
    )

    temperature_violations = int(
        (
            (
                df["temperature_min"]
                > df["temperature_mean"]
            )
            |
            (
                df["temperature_mean"]
                > df["temperature_max"]
            )
        ).sum()
    )

    if duplicate_count != 0:
        raise AssertionError(
            f"{name}: duplicate city-date rows = "
            f"{duplicate_count}"
        )

    if missing_weather_values != 0:
        raise AssertionError(
            f"{name}: missing weather values = "
            f"{missing_weather_values}"
        )

    if temperature_violations != 0:
        raise AssertionError(
            f"{name}: temperature consistency violations = "
            f"{temperature_violations}"
        )

    expected_days = (
        max_date - min_date
    ).days + 1

    expected_rows = (
        expected_locations
        * expected_days
    )

    assert_equal(
        len(df),
        expected_rows,
        f"{name}: total rows based on continuous date coverage",
    )

    rows_per_location = (
        df.groupby(
            "location_id"
        )
        .size()
    )

    if not (
        rows_per_location == expected_days
    ).all():
        raise AssertionError(
            f"{name}: not every location contains "
            f"{expected_days} daily observations"
        )

    date_count_per_location = (
        df.groupby(
            "location_id"
        )["date"]
        .nunique()
    )

    if not (
        date_count_per_location == expected_days
    ).all():
        raise AssertionError(
            f"{name}: date continuity check failed"
        )

    return {
        "rows": int(len(df)),
        "locations": location_count,
        "cities": city_count,
        "minimum_date": str(min_date),
        "maximum_date": str(max_date),
        "days_per_location": int(expected_days),
        "duplicates": duplicate_count,
        "missing_weather_values": missing_weather_values,
        "temperature_consistency_violations": (
            temperature_violations
        ),
        "status": "PASS",
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print("=" * 80)
    print("PHASE 11C — FINAL DATA QUALITY & PROVENANCE AUDIT")
    print("=" * 80)

    report = {
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": "PASS",
        "datasets": {},
        "provenance": {},
    }

    # -------------------------------------------------------------------------
    # 1. LOCATION CATALOGUE
    # -------------------------------------------------------------------------

    locations = pd.read_csv(
        LOCATIONS_FILE
    )

    assert_equal(
        len(locations),
        100,
        "Location catalogue row count",
    )

    assert_equal(
        locations["location_id"].nunique(),
        100,
        "Unique location IDs",
    )

    assert_equal(
        locations["city"].nunique(),
        100,
        "Unique cities",
    )

    assert_equal(
        locations["country"].nunique(),
        80,
        "Unique countries",
    )

    report["datasets"]["locations"] = {
        "rows": 100,
        "cities": 100,
        "countries": 80,
        "duplicate_location_ids": int(
            locations["location_id"]
            .duplicated()
            .sum()
        ),
        "status": "PASS",
    }

    print()
    print("[PASS] Location catalogue")
    print("       100 cities / 80 countries")

    # -------------------------------------------------------------------------
    # 2. CLUSTER DASHBOARD DATASET
    # -------------------------------------------------------------------------

    clusters = pd.read_parquet(
        CLUSTER_FILE
    )

    assert_equal(
        len(clusters),
        100,
        "Cluster dataset rows",
    )

    assert_equal(
        clusters["location_id"].nunique(),
        100,
        "Cluster unique location IDs",
    )

    missing_cluster_columns = [
        column
        for column in CLUSTERING_COLUMNS
        if column not in clusters.columns
    ]

    if missing_cluster_columns:
        raise AssertionError(
            "Missing clustering columns: "
            f"{missing_cluster_columns}"
        )

    if clusters[
        CLUSTERING_COLUMNS
    ].isna().any().any():
        raise AssertionError(
            "Missing values detected in clustering indicators"
        )

    cluster_ids = sorted(
        int(value)
        for value
        in clusters[
            "cluster_id"
        ].unique()
    )

    assert_equal(
        cluster_ids,
        [0, 1, 2, 3],
        "Final cluster IDs",
    )

    cluster_counts = {
        str(int(cluster_id)): int(count)
        for cluster_id, count
        in clusters[
            "cluster_id"
        ]
        .value_counts()
        .sort_index()
        .items()
    }

    expected_cluster_counts = {
        "0": 36,
        "1": 20,
        "2": 34,
        "3": 10,
    }

    assert_equal(
        cluster_counts,
        expected_cluster_counts,
        "Final K=4 cluster sizes",
    )

    report[
        "datasets"
    ][
        "final_clustering"
    ] = {
        "rows": 100,
        "k": 4,
        "cluster_ids": cluster_ids,
        "cluster_sizes": cluster_counts,
        "clustering_variables": CLUSTERING_COLUMNS,
        "status": "PASS",
    }

    print()
    print("[PASS] Final K=4 clustering")
    print(
        f"       Cluster sizes: {cluster_counts}"
    )

    # -------------------------------------------------------------------------
    # 3. DERIVED EDA FEATURE TABLE
    # -------------------------------------------------------------------------

    features = pd.read_parquet(
        CLIMATE_FEATURE_FILE
    )

    assert_equal(
        len(features),
        100,
        "Climate feature rows",
    )

    missing_features = [
        column
        for column in CLIMATE_FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing_features:
        raise AssertionError(
            f"Missing climate features: {missing_features}"
        )

    if features[
        CLIMATE_FEATURE_COLUMNS
    ].isna().any().any():
        raise AssertionError(
            "Missing values found in climate feature table"
        )

    report[
        "datasets"
    ][
        "derived_climate_features"
    ] = {
        "rows": 100,
        "derived_numeric_features": 16,
        "used_for_kmeans": False,
        "role": (
            "EDA and descriptive climate analysis only"
        ),
        "status": "PASS",
    }

    print()
    print("[PASS] Derived climate feature table")
    print(
        "       100 cities / 16 derived EDA features"
    )

    # -------------------------------------------------------------------------
    # 4. PUBLIC HUGGING FACE ERA5
    # -------------------------------------------------------------------------

    era5_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_ERA5_FILE,
        repo_type="dataset",
        token=False,
    )

    era5 = pd.read_parquet(
        era5_path
    )

    era5_result = validate_daily_weather(
        era5,
        name="ERA5 baseline",
        expected_start="2016-01-01",
        expected_end="2025-12-31",
    )

    assert_equal(
        era5_result["rows"],
        365_300,
        "ERA5 total rows",
    )

    assert_equal(
        era5_result["days_per_location"],
        3653,
        "ERA5 observations per city",
    )

    report[
        "datasets"
    ][
        "era5_baseline"
    ] = era5_result

    print()
    print("[PASS] ERA5 climate baseline")
    print(
        f"       {era5_result['rows']:,} rows"
    )
    print(
        "       2016-01-01 → 2025-12-31"
    )
    print(
        "       3,653 observations per city"
    )

    # -------------------------------------------------------------------------
    # 5. PUBLIC HUGGING FACE RECENT IFS
    # -------------------------------------------------------------------------

    ifs_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_IFS_FILE,
        repo_type="dataset",
        token=False,
    )

    ifs = pd.read_parquet(
        ifs_path
    )

    ifs_result = validate_daily_weather(
        ifs,
        name="ECMWF IFS recent layer",
        expected_start="2026-01-01",
        expected_end=None,
    )

    report[
        "datasets"
    ][
        "ecmwf_ifs_recent"
    ] = ifs_result

    print()
    print("[PASS] ECMWF IFS recent layer")
    print(
        f"       {ifs_result['rows']:,} rows"
    )
    print(
        f"       2026-01-01 → "
        f"{ifs_result['maximum_date']}"
    )
    print(
        f"       {ifs_result['days_per_location']} "
        "observations per city"
    )

    # -------------------------------------------------------------------------
    # 6. PROVENANCE / ARCHITECTURE DECLARATION
    # -------------------------------------------------------------------------

    report["provenance"] = {
        "provider": "Open-Meteo",
        "era5": {
            "underlying_model": "ERA5",
            "period": "2016-01-01 to 2025-12-31",
            "purpose": (
                "Frozen long-term climate baseline "
                "and official K=4 clustering"
            ),
            "automatically_retrained": False,
        },
        "ecmwf_ifs": {
            "underlying_model": "ECMWF IFS",
            "period": (
                "2026-01-01 to latest published complete day"
            ),
            "purpose": (
                "Recent historical operational weather layer"
            ),
            "automatically_updated": True,
        },
        "real_time": {
            "storage": "Not permanently stored",
            "purpose": (
                "Current conditions retrieved live "
                "through Open-Meteo"
            ),
        },
        "analytics": {
            "kmeans_engine": "Apache Spark MLlib",
            "final_k": 4,
            "k_selection_range": "2 to 8",
            "pca_role": (
                "Visualization only; PCA was not used "
                "as clustering input"
            ),
            "derived_features_used_for_kmeans": False,
        },
        "storage": {
            "raw": "JSON",
            "analytical": "Apache Parquet",
            "cloud_dataset_store": "Hugging Face Datasets",
        },
        "automation": {
            "scheduler": "GitHub Actions",
            "dashboard_host": (
                "Streamlit Community Cloud"
            ),
        },
    }

    # -------------------------------------------------------------------------
    # WRITE REPORT
    # -------------------------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print("PHASE 11C AUDIT PASSED")
    print("=" * 80)

    print()
    print(
        f"Audit report written to:\n{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()
