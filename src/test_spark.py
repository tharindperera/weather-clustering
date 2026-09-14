import os
import sys

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .master("local[*]")
    .appName("WeatherClusteringTest")
    .getOrCreate()
)

data = [
    ("Colombo", 28.4, 2200),
    ("London", 11.2, 600),
    ("Tokyo", 17.8, 1500),
    ("Cairo", 24.3, 20),
    ("Sydney", 19.1, 1200),
]

columns = ["city", "temperature", "rainfall"]

df = spark.createDataFrame(data, columns)

df.show()

print("Row count:", df.count())

spark.stop()
