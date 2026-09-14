import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["PYTHONFAULTHANDLER"] = "1"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator


def main() -> None:
    spark = (
        SparkSession.builder
        .master("local[2]")
        .config("spark.driver.host", "localhost")
        .config("spark.python.worker.reuse", "true")
        .config("spark.python.worker.faulthandler.enabled", "true")
        .appName("WeatherClusteringEnvironmentTest")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        # -------------------------------------------------
        # 1. Create a small weather-like DataFrame
        # -------------------------------------------------
        data = [
            ("Colombo", 28.4, 2200.0, 78.0),
            ("London", 11.2, 600.0, 75.0),
            ("Tokyo", 17.8, 1500.0, 70.0),
            ("Cairo", 24.3, 20.0, 45.0),
            ("Sydney", 19.1, 1200.0, 68.0),
            ("Mumbai", 27.0, 2200.0, 80.0),
            ("Moscow", 5.5, 700.0, 70.0),
            ("Dubai", 29.0, 100.0, 40.0),
        ]

        columns = ["city", "temperature", "rainfall", "humidity"]

        df = spark.createDataFrame(data, columns)

        print("\n=== 1. SPARK DATAFRAME TEST ===")
        df.show()

        print("Row count:", df.count())

        # -------------------------------------------------
        # 2. Spark SQL test
        # -------------------------------------------------
        print("\n=== 2. SPARK SQL TEST ===")

        df.createOrReplaceTempView("weather")

        sql_result = spark.sql(
            """
            SELECT
                city,
                temperature,
                rainfall,
                humidity
            FROM weather
            WHERE temperature > 20
            ORDER BY temperature DESC
            """
        )

        sql_result.show()

        # -------------------------------------------------
        # 3. Aggregation test
        # -------------------------------------------------
        print("\n=== 3. SPARK AGGREGATION TEST ===")

        summary = (
            df.groupBy()
            .agg(
                F.avg("temperature").alias("avg_temperature"),
                F.avg("rainfall").alias("avg_rainfall"),
                F.avg("humidity").alias("avg_humidity"),
            )
        )

        summary.show()

        # -------------------------------------------------
        # 4. Feature vector
        # -------------------------------------------------
        print("\n=== 4. VECTOR ASSEMBLER TEST ===")

        assembler = VectorAssembler(
            inputCols=["temperature", "rainfall", "humidity"],
            outputCol="features_raw",
        )

        assembled = assembler.transform(df)

        assembled.select("city", "features_raw").show()

        # -------------------------------------------------
        # 5. StandardScaler
        # -------------------------------------------------
        print("\n=== 5. STANDARD SCALER TEST ===")

        scaler = StandardScaler(
            inputCol="features_raw",
            outputCol="features",
            withMean=True,
            withStd=True,
        )

        scaler_model = scaler.fit(assembled)
        scaled = scaler_model.transform(assembled)

        scaled.select("city", "features").show(truncate=False)

        # -------------------------------------------------
        # 6. K-Means
        # -------------------------------------------------
        print("\n=== 6. K-MEANS TEST ===")

        kmeans = KMeans(
            k=3,
            seed=42,
            featuresCol="features",
            predictionCol="cluster",
        )

        model = kmeans.fit(scaled)
        result = model.transform(scaled)

        result.select("city", "cluster").show()

        # -------------------------------------------------
        # 7. Silhouette score
        # -------------------------------------------------
        print("\n=== 7. SILHOUETTE TEST ===")

        evaluator = ClusteringEvaluator(
            featuresCol="features",
            predictionCol="cluster",
        )

        silhouette = evaluator.evaluate(result)

        print(f"Silhouette score: {silhouette:.4f}")

        print("\n======================================")
        print("ALL SPARK / MLLIB TESTS PASSED")
        print("======================================")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
