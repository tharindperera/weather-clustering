from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PROCESSED_DIR, LOCATIONS_FILE

# ---------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------

CLUSTER_FILE = PROCESSED_DIR / "clustering" / "final_clusters_k4.parquet"

LOCATION_FILE = LOCATIONS_FILE

OUTPUT_DIR = PROCESSED_DIR / "clustering"

OUTPUT_FILE = OUTPUT_DIR / "dashboard_clustering_dataset.parquet"
OUTPUT_CSV = OUTPUT_DIR / "dashboard_clustering_dataset.csv"


# ---------------------------------------------------------------------
# Original weather variables used in clustering
# ---------------------------------------------------------------------

WEATHER_COLUMNS = [
    "temperature_mean",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]


def main() -> None:
    print("=" * 80)
    print("BUILD DASHBOARD-READY CLUSTERING DATASET")
    print("=" * 80)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # 1. Load final clustering results
    # -----------------------------------------------------------------

    print()
    print("1. LOADING FINAL K=4 CLUSTER DATA")
    print("-" * 80)

    if not CLUSTER_FILE.exists():
        raise FileNotFoundError(f"Cluster file not found: {CLUSTER_FILE}")

    clusters = pd.read_parquet(CLUSTER_FILE)

    print(f"Rows: {len(clusters):,}")
    print(
        f"Columns: {len(clusters.columns):,}"
    )

    # -----------------------------------------------------------------
    # 2. Load location catalogue
    # -----------------------------------------------------------------

    print()
    print("2. LOADING LOCATION CATALOGUE")
    print("-" * 80)

    if not LOCATION_FILE.exists():
        raise FileNotFoundError(f"Location file not found: {LOCATION_FILE}")

    locations = pd.read_csv(
        LOCATION_FILE
    )

    print(f"Rows: {len(locations):,}")

    # -----------------------------------------------------------------
    # 3. Validate source data
    # -----------------------------------------------------------------

    required_cluster_columns = [
        "location_id",
        "city",
        "country",
        "cluster_id",
        *WEATHER_COLUMNS,
    ]

    required_location_columns = [
        "location_id",
        "city",
        "country",
        "latitude",
        "longitude",
    ]

    missing_cluster_columns = [
        column
        for column in required_cluster_columns
        if column not in clusters.columns
    ]

    missing_location_columns = [
        column
        for column in required_location_columns
        if column not in locations.columns
    ]

    if missing_cluster_columns:
        raise ValueError(
            "Missing cluster columns: "
            f"{missing_cluster_columns}"
        )

    if missing_location_columns:
        raise ValueError(
            "Missing location columns: "
            f"{missing_location_columns}"
        )

    if len(clusters) != 100:
        raise ValueError(
            f"Expected 100 cluster rows, found {len(clusters)}."
        )

    if clusters["location_id"].nunique() != 100:
        raise ValueError(
            "Cluster dataset does not contain 100 unique location IDs."
        )

    # -----------------------------------------------------------------
    # 4. Select location metadata
    # -----------------------------------------------------------------

    location_metadata = locations[
        [
            "location_id",
            "latitude",
            "longitude",
        ]
    ].copy()

    if location_metadata["location_id"].duplicated().any():
        raise ValueError(
            "Duplicate location IDs found in location catalogue."
        )

    # -----------------------------------------------------------------
    # 5. Join coordinates to cluster results
    # -----------------------------------------------------------------

    print()
    print("3. JOINING GEOGRAPHIC COORDINATES")
    print("-" * 80)

    df = clusters[
        [
            "location_id",
            "city",
            "country",
            "cluster_id",
            *WEATHER_COLUMNS,
        ]
    ].merge(
        location_metadata,
        on="location_id",
        how="left",
        validate="one_to_one",
    )

    # -----------------------------------------------------------------
    # 6. Validate coordinate join
    # -----------------------------------------------------------------

    if df["latitude"].isna().any():
        missing = df.loc[
            df["latitude"].isna(),
            ["location_id", "city"],
        ]

        raise ValueError(
            "Missing latitude values after join:\n"
            f"{missing.to_string(index=False)}"
        )

    if df["longitude"].isna().any():
        missing = df.loc[
            df["longitude"].isna(),
            ["location_id", "city"],
        ]

        raise ValueError(
            "Missing longitude values after join:\n"
            f"{missing.to_string(index=False)}"
        )

    if df[WEATHER_COLUMNS].isna().any().any():
        raise ValueError(
            "Missing weather values detected."
        )

    if df["cluster_id"].isna().any():
        raise ValueError(
            "Missing cluster assignments detected."
        )

    # -----------------------------------------------------------------
    # 7. Rename fields for dashboard readability
    # -----------------------------------------------------------------

    df = df.rename(
        columns={
            "temperature_mean": "temperature_c",
            "precipitation_sum": "rainfall_mm_per_day",
            "relative_humidity_mean": "humidity_percent",
            "wind_speed_mean": "wind_speed_kmh",
            "surface_pressure_mean": "pressure_hpa",
        }
    )

    # -----------------------------------------------------------------
    # 8. Sort and arrange final schema
    # -----------------------------------------------------------------

    df = df[
        [
            "location_id",
            "city",
            "country",
            "latitude",
            "longitude",
            "cluster_id",
            "temperature_c",
            "rainfall_mm_per_day",
            "humidity_percent",
            "wind_speed_kmh",
            "pressure_hpa",
        ]
    ].sort_values(
        ["cluster_id", "city"]
    ).reset_index(
        drop=True
    )

    # -----------------------------------------------------------------
    # 9. Final validation
    # -----------------------------------------------------------------

    print()
    print("4. FINAL VALIDATION")
    print("-" * 80)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Unique cities: {df['city'].nunique():,}"
    )

    print(
        f"Unique clusters: {df['cluster_id'].nunique():,}"
    )

    print()
    print("Cluster counts:")

    print(
        df["cluster_id"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print("Final columns:")

    for column in df.columns:
        print(
            f"  - {column}"
        )

    if len(df) != 100:
        raise ValueError(
            "Final dataset does not contain exactly 100 rows."
        )

    if df["city"].nunique() != 100:
        raise ValueError(
            "Final dataset does not contain 100 unique cities."
        )

    if not df["cluster_id"].isin(
        [0, 1, 2, 3]
    ).all():
        raise ValueError(
            "Unexpected cluster IDs found."
        )

    # -----------------------------------------------------------------
    # 10. Save
    # -----------------------------------------------------------------

    print()
    print("5. SAVING DASHBOARD DATASET")
    print("-" * 80)

    df.to_parquet(
        OUTPUT_FILE,
        index=False,
        engine="pyarrow",
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print()
    print("Parquet:")
    print(
        f"  {OUTPUT_FILE}"
    )

    print()
    print("CSV:")
    print(
        f"  {OUTPUT_CSV}"
    )

    print()
    print("=" * 80)
    print("DASHBOARD DATASET CREATED AND VALIDATED")
    print("=" * 80)


if __name__ == "__main__":
    main()
