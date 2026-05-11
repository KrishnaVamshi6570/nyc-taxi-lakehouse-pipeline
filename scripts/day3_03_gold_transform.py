from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE = "gs://gcp-pyspark-lakehouse-cdebatch52-486907"
SILVER_PATH = BASE + "/silver/nyc_taxi_delta"
GOLD_BASE   = BASE + "/gold"

spark = SparkSession.builder \
    .appName("day3-gold-transform") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.3.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.sql.parquet.mergeSchema", "false") \
    .getOrCreate()

print("=== Reading Silver Delta table ===")
silver = spark.read.format("delta").load(SILVER_PATH)
silver.cache()
print(f"Silver row count: {silver.count():,}")

# ── Gold 1: Hourly Stats ─────────────────────────────────────────────────────
# Grain: pickup_hour x pickup_dayofweek
# Purpose: time-based feature enrichment for Day 4 ML tip prediction
print("\n=== Building gold_hourly_stats ===")
hourly_stats = silver.groupBy("pickup_hour", "pickup_dayofweek").agg(
    F.count("*").alias("trip_count"),
    F.round(F.avg("fare_amount"), 4).alias("avg_fare"),
    F.round(F.avg("tip_pct"), 4).alias("avg_tip_pct"),
    F.round(F.avg("trip_duration_minutes"), 4).alias("avg_trip_duration_minutes"),
    F.round(F.avg("speed_mph"), 4).alias("avg_speed_mph"),
    F.round(F.avg("tip_amount"), 4).alias("avg_tip_amount")
).orderBy("pickup_dayofweek", "pickup_hour")

hourly_path = GOLD_BASE + "/gold_hourly_stats"
hourly_stats.write.format("delta").mode("overwrite").save(hourly_path)
print(f"gold_hourly_stats written: {hourly_stats.count()} rows -> {hourly_path}")

# ── Gold 2: Location Stats ───────────────────────────────────────────────────
# Grain: pu_location_id
# Purpose: location-level tip baseline — joinable feature for Day 4 ML model
print("\n=== Building gold_location_stats ===")
location_stats = silver.groupBy("pu_location_id").agg(
    F.count("*").alias("trip_count"),
    F.round(F.avg("fare_amount"), 4).alias("avg_fare"),
    F.round(F.avg("tip_pct"), 4).alias("avg_tip_pct"),
    F.round(F.avg("trip_duration_minutes"), 4).alias("avg_trip_duration_minutes"),
    F.round(F.avg("tip_amount"), 4).alias("avg_tip_amount"),
    F.round(F.avg("is_airport_trip"), 4).alias("airport_trip_rate")
).orderBy("pu_location_id")

location_path = GOLD_BASE + "/gold_location_stats"
location_stats.write.format("delta").mode("overwrite").save(location_path)
print(f"gold_location_stats written: {location_stats.count()} rows -> {location_path}")

# ── Gold 3: Monthly Stats ────────────────────────────────────────────────────
# Derive year and pickup_month from pickup_datetime if not present
from pyspark.sql import functions as F
if "pickup_year" not in silver.columns:
    silver = silver.withColumn("pickup_year", F.year("pickup_datetime"))
if "pickup_month" not in silver.columns:
    silver = silver.withColumn("pickup_month", F.month("pickup_datetime"))

# Grain: year x month
# Purpose: revenue summary + drift baseline for Day 6 Looker Studio
print("\n=== Building gold_monthly_stats ===")
monthly_stats = silver.groupBy("pickup_year", "pickup_month").agg(
    F.count("*").alias("total_trips"),
    F.round(F.sum("fare_amount"), 2).alias("total_revenue"),
    F.round(F.sum("tip_amount"), 2).alias("total_tips"),
    F.round(F.avg("fare_amount"), 4).alias("avg_fare"),
    F.round(F.avg("tip_pct"), 4).alias("avg_tip_pct"),
    F.round(F.avg("speed_mph"), 4).alias("avg_speed_mph"),
    F.round(F.avg("trip_duration_minutes"), 4).alias("avg_trip_duration_minutes")
).orderBy("pickup_year", "pickup_month")

monthly_path = GOLD_BASE + "/gold_monthly_stats"
monthly_stats.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("pickup_year", "pickup_month") \
    .save(monthly_path)
print(f"gold_monthly_stats written: {monthly_stats.count()} rows -> {monthly_path}")

print("\n=== Gold transform complete ===")
spark.stop()
