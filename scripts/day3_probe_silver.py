from pyspark.sql import SparkSession

BASE = "gs://gcp-pyspark-lakehouse-cdebatch52-486907"

spark = SparkSession.builder \
    .appName("day3-silver-probe") \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.3.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.sql.parquet.mergeSchema", "false") \
    .getOrCreate()

silver_path = BASE + "/silver/nyc_taxi_delta"
df = spark.read.format("delta").load(silver_path)

print("=== SILVER TABLE PROBE ===")
print(f"Row count      : {df.count():,}")
print(f"Partition cols : year, month")
print(f"Columns        : {df.columns}")
df.printSchema()
df.show(3, truncate=False)

spark.stop()
