from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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

OUTPUT_DIR = PROCESSED_DIR / "clustering" / "model_selection"

RESULTS_FILE = OUTPUT_DIR / "kmeans_model_selection.csv"


# ---------------------------------------------------------------------
# Original weather variables used for clustering
# ---------------------------------------------------------------------

WEATHER_COLUMNS = [
    "temperature_mean",
    "precipitation_sum",
    "relative_humidity_mean",
    "wind_speed_mean",
    "surface_pressure_mean",
]


# Candidate K values
K_VALUES = list(range(2, 9))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    print("=" * 80)
    print("PYSPARK K-MEANS MODEL SELECTION")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Start Spark
    # -----------------------------------------------------------------

    spark = (
        SparkSession.builder
        .appName("WeatherPatternKMeansModelSelection")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        # -------------------------------------------------------------
        # 1. Load clustering input
        # -------------------------------------------------------------

        print()
        print("1. LOADING CLUSTERING INPUT")
        print("-" * 80)

        df = spark.read.parquet(str(INPUT_FILE))

        row_count = df.count()

        print(f"Rows: {row_count:,}")

        if row_count != 100:
            raise ValueError(
                f"Expected 100 cities, found {row_count}."
            )

        city_count = df.select("city").distinct().count()

        if city_count != 100:
            raise ValueError(
                f"Expected 100 unique cities, found {city_count}."
            )

        # -------------------------------------------------------------
        # Validate required columns
        # -------------------------------------------------------------

        missing_columns = [
            column
            for column in WEATHER_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required weather columns: {missing_columns}"
            )

        # -------------------------------------------------------------
        # Check missing values
        # -------------------------------------------------------------

        missing_counts = df.select(
            [
                F.sum(
                    F.when(F.col(column).isNull(), 1).otherwise(0)
                ).alias(column)
                for column in WEATHER_COLUMNS
            ]
        ).collect()[0].asDict()

        total_missing = sum(
            int(value or 0)
            for value in missing_counts.values()
        )

        if total_missing != 0:
            raise ValueError(
                f"Missing weather values detected: {missing_counts}"
            )

        print("Input validation: PASSED")

        # -------------------------------------------------------------
        # 2. Assemble five original variables
        # -------------------------------------------------------------

        print()
        print("2. BUILDING WEATHER FEATURE VECTOR")
        print("-" * 80)

        assembler = VectorAssembler(
            inputCols=WEATHER_COLUMNS,
            outputCol="weather_features_raw",
        )

        assembled_df = assembler.transform(df).select(
            "location_id",
            "city",
            "city_ascii",
            "country",
            "iso2",
            "iso3",
            "weather_features_raw",
        )

        # -------------------------------------------------------------
        # 3. Standardize with Spark MLlib
        # -------------------------------------------------------------

        print()
        print("3. STANDARDIZING WEATHER FEATURES")
        print("-" * 80)

        scaler = StandardScaler(
            inputCol="weather_features_raw",
            outputCol="features",
            withStd=True,
            withMean=True,
        )

        scaler_model = scaler.fit(assembled_df)

        scaled_df = (
            scaler_model
            .transform(assembled_df)
            .select(
                "location_id",
                "city",
                "city_ascii",
                "country",
                "iso2",
                "iso3",
                "features",
            )
            .cache()
        )

        scaled_count = scaled_df.count()

        if scaled_count != 100:
            raise ValueError(
                f"Expected 100 standardized rows, found {scaled_count}."
            )

        print(
            "Spark StandardScaler completed successfully."
        )

        # -------------------------------------------------------------
        # 4. Evaluate candidate K values
        # -------------------------------------------------------------

        print()
        print("4. K-MEANS MODEL SELECTION")
        print("-" * 80)

        evaluator = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="features",
            metricName="silhouette",
            distanceMeasure="squaredEuclidean",
        )

        results = []

        print()
        print(
            f"{'K':>3} "
            f"{'SSE / Training Cost':>22} "
            f"{'Silhouette':>15} "
            f"{'Min Cluster':>13} "
            f"{'Max Cluster':>13}"
        )
        print("-" * 72)

        for k in K_VALUES:
            print(f"Running K={k}...", flush=True)

            kmeans = KMeans(
                k=k,
                seed=42,
                featuresCol="features",
                predictionCol="prediction",
                maxIter=100,
                tol=1e-4,
            )

            model = kmeans.fit(scaled_df)

            predictions = model.transform(scaled_df)

            # Spark MLlib K-Means training cost = within-cluster
            # sum of squared errors for the fitted model.
            training_cost = model.summary.trainingCost

            silhouette = evaluator.evaluate(predictions)

            cluster_sizes = [
                int(row["count"])
                for row in (
                    predictions
                    .groupBy("prediction")
                    .count()
                    .orderBy("prediction")
                    .collect()
                )
            ]

            if not cluster_sizes:
                raise ValueError(
                    f"No clusters produced for K={k}."
                )

            min_cluster_size = min(cluster_sizes)
            max_cluster_size = max(cluster_sizes)

            result = {
                "k": k,
                "training_cost": float(training_cost),
                "silhouette": float(silhouette),
                "min_cluster_size": min_cluster_size,
                "max_cluster_size": max_cluster_size,
                "cluster_sizes": ",".join(
                    str(size)
                    for size in cluster_sizes
                ),
            }

            results.append(result)

            print(
                f"{k:>3} "
                f"{training_cost:>22.6f} "
                f"{silhouette:>15.6f} "
                f"{min_cluster_size:>13} "
                f"{max_cluster_size:>13}"
            )

        # -------------------------------------------------------------
        # 5. Save results
        # -------------------------------------------------------------

        print()
        print("5. MODEL-SELECTION RESULTS")
        print("-" * 80)

        results_df = spark.createDataFrame(results)

        results_df = results_df.orderBy("k")

        results_df.show(
            truncate=False
        )

        # Save CSV using pandas export from Spark DataFrame (avoids Hadoop winutils permissions on Windows)
        pd_results = results_df.toPandas()
        pd_results.to_csv(RESULTS_FILE, index=False)

        print()
        print(f"Results written to:")
        print(f"  {RESULTS_FILE}")

        # -------------------------------------------------------------
        # 6. Basic automatic checks
        # -------------------------------------------------------------

        if len(results) != len(K_VALUES):
            raise ValueError(
                "Not all candidate K values were evaluated."
            )

        best_by_silhouette = max(
            results,
            key=lambda row: row["silhouette"],
        )

        lowest_cost = min(
            results,
            key=lambda row: row["training_cost"],
        )

        print()
        print("Highest silhouette:")
        print(
            f"  K={best_by_silhouette['k']}, "
            f"silhouette={best_by_silhouette['silhouette']:.6f}"
        )

        print()
        print("Lowest training cost:")
        print(
            f"  K={lowest_cost['k']}, "
            f"training_cost={lowest_cost['training_cost']:.6f}"
        )

        print()
        print("=" * 80)
        print("K-MEANS MODEL SELECTION COMPLETE")
        print("=" * 80)

    finally:
        scaled_df.unpersist(blocking=False) if "scaled_df" in locals() else None
        spark.stop()


if __name__ == "__main__":
    main()
