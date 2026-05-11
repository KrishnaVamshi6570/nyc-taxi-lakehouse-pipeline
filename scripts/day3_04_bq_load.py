from pyspark.sql import SparkSession

BASE      = "gs://gcp-pyspark-lakehouse-cdebatch52-486907"
GOLD_BASE = BASE + "/gold"
BQ_DATASET = "cdebatch52-486907.lakehouse_gold"
TEMP_BUCKET = "gcp-pyspark-lakehouse-cdebatch52-486907"

spark = SparkSession.builder \
    .appName("day3-bq-load") \
    .config("spark.jars.packages",
            "io.delta:delta-core_2.12:2.3.0,"
            "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.32.2") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.sql.parquet.mergeSchema", "false") \
    .getOrCreate()

tables = [
    ("gold_hourly_stats",   GOLD_BASE + "/gold_hourly_stats"),
    ("gold_location_stats", GOLD_BASE + "/gold_location_stats"),
    ("gold_monthly_stats",  GOLD_BASE + "/gold_monthly_stats"),
]

for bq_table, delta_path in tables:
    print(f"\n=== Loading {bq_table} -> BigQuery ===")
    df = spark.read.format("delta").load(delta_path)
    print(f"  Row count: {df.count():,}")
    df.write \
        .format("bigquery") \
        .option("table", BQ_DATASET + "." + bq_table) \
        .option("temporaryGcsBucket", TEMP_BUCKET) \
        .mode("overwrite") \
        .save()
    print(f"  Written to BigQuery: {BQ_DATASET}.{bq_table}")

print("\n=== BQ load complete ===")
spark.stop()
