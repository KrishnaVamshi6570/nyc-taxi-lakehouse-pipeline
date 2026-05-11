"""
Day 2 — Silver Transform (Bronze -> Silver Delta Lake)
gcp-pyspark-lakehouse-pipeline
- Reads Bronze parquet month by month (month=01 .. month=12)
- Enforces canonical schema, cleans, derives features
- Writes Delta table partitioned by pickup_month
"""

import argparse
from datetime import datetime

from delta import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, TimestampType
)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True)
parser.add_argument("--mode", default="overwrite", choices=["overwrite", "merge"])
args = parser.parse_args()

BUCKET      = args.bucket
BRONZE_BASE = "gs://" + BUCKET + "/bronze/nyc_taxi/year=2023"
SILVER_PATH = "gs://" + BUCKET + "/silver/nyc_taxi_delta"

# ── Spark Session ──────────────────────────────────────────────────────────────
spark = (
    SparkSession.builder
    .appName("Day2-SilverTransform")
    .config("spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.parquet.enableVectorizedReader", "false")
    .config("spark.sql.parquet.mergeSchema", "false")
    .config("spark.databricks.delta.optimizeWrite.enabled", "true")
    .config("spark.databricks.delta.autoCompact.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

print("Silver path : " + SILVER_PATH)
print("Write mode  : " + args.mode)

# ── Canonical schema map ───────────────────────────────────────────────────────
SCHEMA_MAP = {
    "VendorID":              ("vendor_id",              IntegerType()),
    "tpep_pickup_datetime":  ("pickup_datetime",        TimestampType()),
    "tpep_dropoff_datetime": ("dropoff_datetime",       TimestampType()),
    "passenger_count":       ("passenger_count",        IntegerType()),
    "trip_distance":         ("trip_distance",          DoubleType()),
    "RatecodeID":            ("ratecode_id",            IntegerType()),
    "store_and_fwd_flag":    ("store_and_fwd_flag",     StringType()),
    "PULocationID":          ("pu_location_id",         IntegerType()),
    "DOLocationID":          ("do_location_id",         IntegerType()),
    "payment_type":          ("payment_type",           IntegerType()),
    "fare_amount":           ("fare_amount",            DoubleType()),
    "extra":                 ("extra",                  DoubleType()),
    "mta_tax":               ("mta_tax",                DoubleType()),
    "tip_amount":            ("tip_amount",             DoubleType()),
    "tolls_amount":          ("tolls_amount",           DoubleType()),
    "improvement_surcharge": ("improvement_surcharge",  DoubleType()),
    "total_amount":          ("total_amount",           DoubleType()),
    "congestion_surcharge":  ("congestion_surcharge",   DoubleType()),
    "airport_fee":           ("airport_fee",            DoubleType()),
}

def enforce_schema(df):
    exprs = []
    for raw_col, (silver_col, cast_type) in SCHEMA_MAP.items():
        if raw_col in df.columns:
            exprs.append(F.col(raw_col).cast(cast_type).alias(silver_col))
        else:
            exprs.append(F.lit(None).cast(cast_type).alias(silver_col))
    return df.select(exprs)

def clean(df):
    return (
        df
        .filter(F.col("pickup_datetime").isNotNull())
        .filter(F.col("dropoff_datetime").isNotNull())
        .filter(F.col("fare_amount") > 0)
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("passenger_count").isNotNull())
        .filter(F.col("passenger_count").between(1, 8))
        .withColumn("tip_amount",   F.greatest(F.col("tip_amount"),   F.lit(0.0)))
        .withColumn("total_amount", F.greatest(F.col("total_amount"), F.lit(0.0)))
        .filter(F.col("dropoff_datetime") > F.col("pickup_datetime"))
    )

def derive_features(df):
    return (
        df
        .withColumn(
            "trip_duration_minutes",
            (F.unix_timestamp("dropoff_datetime") -
             F.unix_timestamp("pickup_datetime")) / 60.0
        )
        .withColumn(
            "speed_mph",
            F.when(
                F.col("trip_duration_minutes") > 0.5,
                F.col("trip_distance") / (F.col("trip_duration_minutes") / 60.0)
            ).otherwise(F.lit(None).cast(DoubleType()))
        )
        .withColumn(
            "tip_pct",
            F.when(
                F.col("fare_amount") > 0,
                F.round(F.col("tip_amount") / F.col("fare_amount") * 100.0, 2)
            ).otherwise(F.lit(0.0))
        )
        .withColumn("pickup_hour",      F.hour("pickup_datetime"))
        .withColumn("pickup_dayofweek", F.dayofweek("pickup_datetime"))
        .withColumn("pickup_month",     F.month("pickup_datetime"))
        .withColumn(
            "is_airport_trip",
            F.col("ratecode_id").isin([2, 3]).cast(IntegerType())
        )
        .withColumn(
            "row_id",
            F.md5(F.concat_ws("|",
                F.col("pickup_datetime").cast(StringType()),
                F.col("dropoff_datetime").cast(StringType()),
                F.col("pu_location_id").cast(StringType()),
                F.col("do_location_id").cast(StringType()),
                F.col("fare_amount").cast(StringType()),
            ))
        )
        .withColumn("silver_loaded_utc", F.lit(datetime.utcnow().isoformat()))
    )

# ── Read all 12 months ─────────────────────────────────────────────────────────
all_df = None

for mm in ["01","02","03","04","05","06","07","08","09","10","11","12"]:
    path = BRONZE_BASE + "/month=" + mm
    print("Reading: " + path)
    try:
        raw = spark.read.parquet(path)
        silver = derive_features(clean(enforce_schema(raw)))
        if all_df is None:
            all_df = silver
        else:
            all_df = all_df.unionByName(silver, allowMissingColumns=True)
    except Exception as e:
        print("  WARN skipping " + mm + ": " + str(e))

if all_df is None:
    raise RuntimeError("No Bronze data could be read — aborting.")

all_df = all_df.repartition(48, "pickup_month")

# ── Write Delta ────────────────────────────────────────────────────────────────
if args.mode == "overwrite" or not DeltaTable.isDeltaTable(spark, SILVER_PATH):
    print("Writing Silver (OVERWRITE) -> " + SILVER_PATH)
    (
        all_df.write
        .format("delta")
        .mode("overwrite")
        .partitionBy("pickup_month")
        .option("overwriteSchema", "true")
        .save(SILVER_PATH)
    )
    print("Silver written.")
else:
    print("Merging into Silver Delta table...")
    dt = DeltaTable.forPath(spark, SILVER_PATH)
    (
        dt.alias("e")
        .merge(all_df.alias("i"), "e.row_id = i.row_id")
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
    print("Merge complete.")

# ── Validate ───────────────────────────────────────────────────────────────────
print("\n── Row counts per pickup_month ──")
spark.read.format("delta").load(SILVER_PATH) \
    .groupBy("pickup_month").count().orderBy("pickup_month").show()

print("Day 2 Silver DONE. Table at: " + SILVER_PATH)
spark.stop()
