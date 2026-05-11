from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE        = "gs://gcp-pyspark-lakehouse-cdebatch52-486907"
SILVER_PATH = BASE + "/silver/nyc_taxi_delta"
GOLD_BASE   = BASE + "/gold"

spark = SparkSession.builder \
    .appName("day3-gold-monthly") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.3.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.sql.parquet.mergeSchema", "false") \
    .getOrCreate()

silver = spark.read.format("delta").load(SILVER_PATH)

# Print actual column names so we know exactly what's available
print("Silver columns:", silver.columns)

# Derive year safely from pickup_datetime, group by year + pickup_month
silver2 = silver.withColumn("pickup_year", F.year("pickup_datetime"))

# FIX: Filter out bad-year rows (corrupt Silver records with years like 2001, 2003, 2009)
silver2 = silver2.filter(F.col("pickup_year") >= 2020)
print(f"Rows after year filter (>= 2020): {silver2.count()}")

monthly_stats = silver2.groupBy("pickup_year", "pickup_month").agg(
    F.count("*").alias("total_trips"),
    F.round(F.sum("fare_amount"), 2).alias("total_revenue"),
    F.round(F.sum("tip_amount"), 2).alias("total_tips"),
    F.round(F.avg("fare_amount"), 2).alias("avg_fare"),
    F.round(F.avg("tip_pct"), 4).alias("avg_tip_pct"),
    F.round(F.avg("speed_mph"), 4).alias("avg_speed_mph"),
    F.round(F.avg("trip_duration_minutes"), 4).alias("avg_trip_duration_minutes")
).orderBy("pickup_year", "pickup_month")

monthly_path = GOLD_BASE + "/gold_monthly_stats"
monthly_stats.write.format("delta") \
    .mode("overwrite") \
    .partitionBy("pickup_year", "pickup_month") \
    .save(monthly_path)

row_count = monthly_stats.count()
print(f"gold_monthly_stats written: {row_count} rows -> {monthly_path}")
monthly_stats.show(12, truncate=False)

spark.stop()
