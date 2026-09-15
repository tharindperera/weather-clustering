from __future__ import annotations

import os
import sys
import json
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import PROCESSED_DIR

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

INPUT_FILE = PROCESSED_DIR / "features" / "clustering_input_original5.parquet"
OUTPUT_DIR = PROCESSED_DIR / "clustering"
FINAL_MODEL_DIR = OUTPUT_DIR / "final_model"

FINAL_CLUSTERS_PARQUET = OUTPUT_DIR / "final_clusters_k4.parquet"
FINAL_CLUSTERS_CSV = OUTPUT_DIR / "final_clusters_k4.csv"
STABILITY_CSV = FINAL_MODEL_DIR / "kmeans_k4_stability.csv"
CENTROIDS_STANDARDIZED_CSV = FINAL_MODEL_DIR / "centroids_standardized.csv"
CENTROIDS_ORIGINAL_CSV = FINAL_MODEL_DIR / "centroids_original.csv"
CLUSTER_PROFILES_JSON = FINAL_MODEL_DIR / "cluster_profiles.json"

WEATHER_COLUMNS = [
    "temperature_mean",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]

SEEDS = [42, 100, 2024, 777, 999]
K_TARGET = 4


def main() -> None:
    print("=" * 80)
    print("PYSPARK K-MEANS FINALIZATION & STABILITY CHECK (K=4)")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName("WeatherPatternKMeansFinalization")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        # 1. Load input dataset
        print()
        print("1. LOADING CLUSTERING INPUT")
        print("-" * 80)
        df = spark.read.parquet(str(INPUT_FILE))
        city_count = df.count()
        print(f"Loaded {city_count} cities with 5 weather variables.")

        # 2. Vector Assembly & Standardization
        print()
        print("2. ASSEMBLING & STANDARDIZING FEATURES")
        print("-" * 80)
        assembler = VectorAssembler(inputCols=WEATHER_COLUMNS, outputCol="raw_features")
        assembled_df = assembler.transform(df)

        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withMean=True,
            withStd=True,
        )
        scaler_model = scaler.fit(assembled_df)
        scaled_df = scaler_model.transform(assembled_df).cache()

        # 3. Stability Check Across Random Seeds
        print()
        print("3. STABILITY CHECK ACROSS RANDOM SEEDS (K=4)")
        print("-" * 80)

        evaluator = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean",
        )

        stability_results = []
        best_seed = None
        best_silhouette = -1.0
        best_model = None
        best_predictions = None

        for seed in SEEDS:
            kmeans = KMeans(
                k=K_TARGET,
                seed=seed,
                featuresCol="features",
                predictionCol="prediction",
                maxIter=100,
                tol=1e-4,
            )
            model = kmeans.fit(scaled_df)
            predictions = model.transform(scaled_df)

            silhouette = evaluator.evaluate(predictions)
            training_cost = model.summary.trainingCost

            sizes_dict = predictions.groupBy("prediction").count().toPandas().set_index("prediction")["count"].to_dict()
            sorted_sizes = [sizes_dict.get(i, 0) for i in range(K_TARGET)]

            stability_results.append({
                "seed": seed,
                "silhouette": round(silhouette, 6),
                "training_cost": round(training_cost, 6),
                "cluster_sizes": str(sorted_sizes),
            })

            print(f"  Seed {seed:4d} | Silhouette: {silhouette:.6f} | WSSSE: {training_cost:8.4f} | Sizes: {sorted_sizes}")

            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_seed = seed
                best_model = model
                best_predictions = predictions

        # Save stability check results
        stability_df = pd.DataFrame(stability_results)
        stability_df.to_csv(STABILITY_CSV, index=False)
        print()
        print(f"  Selected optimal seed: {best_seed} (Silhouette: {best_silhouette:.6f})")
        print(f"  Stability summary exported to: {STABILITY_CSV}")

        # 4. Standardized Cluster Centroids
        print()
        print("4. STANDARDIZED CLUSTER CENTROIDS (z-scores)")
        print("-" * 80)
        centroids = best_model.clusterCenters()
        std_centroids_df = pd.DataFrame(centroids, columns=[f"{c}_z" for c in WEATHER_COLUMNS])
        std_centroids_df.insert(0, "cluster_id", range(K_TARGET))
        print(std_centroids_df.round(4).to_string(index=False))
        std_centroids_df.to_csv(CENTROIDS_STANDARDIZED_CSV, index=False)

        # 5. City Cluster Assignments
        print()
        print("5. ATTACHING CLUSTER ASSIGNMENTS TO CITIES")
        print("-" * 80)

        final_spark_df = best_predictions.withColumnRenamed("prediction", "cluster_id")
        
        output_pd = final_spark_df.select(
            "location_id", "city", "city_ascii", "country", "iso2", "iso3",
            "cluster_id", *WEATHER_COLUMNS
        ).toPandas()

        output_pd = output_pd.sort_values(by=["cluster_id", "city"]).reset_index(drop=True)

        output_pd.to_parquet(FINAL_CLUSTERS_PARQUET, index=False, engine="pyarrow")
        output_pd.to_csv(FINAL_CLUSTERS_CSV, index=False)

        print(f"  Final clusters exported to:")
        print(f"    - {FINAL_CLUSTERS_PARQUET}")
        print(f"    - {FINAL_CLUSTERS_CSV}")

        # 6. Original-Unit Cluster Profiles
        print()
        print("6. ORIGINAL-UNIT CLUSTER PROFILES")
        print("-" * 80)

        orig_summary_rows = []
        for cluster_id, group in output_pd.groupby("cluster_id"):
            size = len(group)
            row = {"cluster_id": cluster_id, "size": size}
            for col in WEATHER_COLUMNS:
                row[f"{col}_mean"] = round(group[col].mean(), 2)
                row[f"{col}_std"] = round(group[col].std(), 2)
                row[f"{col}_min"] = round(group[col].min(), 2)
                row[f"{col}_max"] = round(group[col].max(), 2)
            orig_summary_rows.append(row)

        orig_summary_df = pd.DataFrame(orig_summary_rows)
        orig_summary_df.to_csv(CENTROIDS_ORIGINAL_CSV, index=False)

        display_summary = orig_summary_df[[
            "cluster_id", "size",
            "temperature_mean_mean", "precipitation_sum_mean",
            "relative_humidity_mean_mean", "wind_speed_mean_mean",
            "surface_pressure_mean_mean"
        ]].rename(columns={
            "temperature_mean_mean": "temp_mean_°C",
            "precipitation_sum_mean": "precip_sum_mm",
            "relative_humidity_mean_mean": "humidity_mean_%",
            "wind_speed_mean_mean": "wind_speed_kmh",
            "surface_pressure_mean_mean": "pressure_hPa",
        })

        print(display_summary.to_string(index=False))

        # 7. Cluster Membership Lists
        print()
        print("7. CLUSTER MEMBERSHIP LISTS")
        print("-" * 80)

        cluster_profiles_dict = {}

        for cid in range(K_TARGET):
            cluster_cities = output_pd[output_pd["cluster_id"] == cid]
            city_list = cluster_cities[["city", "country"]].apply(lambda x: f"{x['city']} ({x['country']})", axis=1).tolist()
            
            std_row = std_centroids_df[std_centroids_df["cluster_id"] == cid].iloc[0].drop("cluster_id").to_dict()
            orig_row = orig_summary_df[orig_summary_df["cluster_id"] == cid].iloc[0].to_dict()

            print(f"\n--- CLUSTER {cid} (n = {len(cluster_cities)}) ---")
            print(f"  Original Means: Temp={orig_row['temperature_mean_mean']}°C | Precip={orig_row['precipitation_sum_mean']}mm | Humidity={orig_row['relative_humidity_mean_mean']}% | Wind={orig_row['wind_speed_mean_mean']}km/h | Press={orig_row['surface_pressure_mean_mean']}hPa")
            print(f"  Cities: {', '.join(city_list)}")

            cluster_profiles_dict[f"cluster_{cid}"] = {
                "cluster_id": int(cid),
                "size": len(cluster_cities),
                "standardized_centroids": {k: round(float(v), 4) for k, v in std_row.items()},
                "original_unit_means": {
                    "temperature_mean": float(orig_row["temperature_mean_mean"]),
                    "precipitation_sum": float(orig_row["precipitation_sum_mean"]),
                    "relative_humidity_mean": float(orig_row["relative_humidity_mean_mean"]),
                    "wind_speed_mean": float(orig_row["wind_speed_mean_mean"]),
                    "surface_pressure_mean": float(orig_row["surface_pressure_mean_mean"]),
                },
                "cities": city_list
            }

        with open(CLUSTER_PROFILES_JSON, "w", encoding="utf-8") as f:
            json.dump(cluster_profiles_dict, f, indent=2)

        print()
        print("=" * 80)
        print("FINAL K-MEANS CLUSTERING (K=4) COMPLETE AND VALIDATED")
        print("=" * 80)

    finally:
        scaled_df.unpersist(blocking=False) if "scaled_df" in locals() else None
        spark.stop()


if __name__ == "__main__":
    main()
