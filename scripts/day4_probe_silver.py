"""
Day 4 — Probe Silver Delta Table
Run this ONCE. No need to rerun if it passes.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BUCKET      = "gcp-pyspark-lakehouse-cdebatch52-486907"
SILVER_PATH = f"gs://{BUCKET}/silver/nyc_taxi_delta"

spark = (
    SparkSession.builder
    .appName("day4_probe_silver")
    .config("spark.sql.shuffle.partitions", "200")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("STEP 1 — Probing Silver Delta table")
print("=" * 60)

silver = spark.read.format("delta").load(SILVER_PATH)
silver = silver.withColumn("pickup_year", F.year("pickup_datetime"))

total = silver.count()
print(f"Total Silver rows (all years): {total:,}")

print("\nYear distribution:")
silver.groupBy("pickup_year").count().orderBy("pickup_year").show()

silver_2023 = silver.filter(F.col("pickup_year") == 2023)
rows_2023 = silver_2023.count()
print(f"Rows for 2023: {rows_2023:,}")

print("\nSchema:")
silver_2023.printSchema()

print("\ntip_pct stats:")
silver_2023.select("tip_pct").summary(
    "count","mean","stddev","min","25%","75%","max"
).show()

print("\nSilver probe PASSED. Proceed to ML script.")
spark.stop()
