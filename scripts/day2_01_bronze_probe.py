"""
Day 2 — Step 1: Bronze Layer Data Probe
gcp-pyspark-lakehouse-pipeline

Purpose:
  Before writing the Silver transform, probe the Bronze parquet files to:
    1. Confirm column names + dtypes across all 12 months
    2. Count nulls per column per month
    3. Find range/outlier stats for key numeric columns
    4. Detect schema drift between months (any column added/removed/retyped)
    5. Emit a JSON summary to GCS for reference

Run on Dataproc (no Delta needed — reads raw Bronze parquet):
  gcloud dataproc jobs submit pyspark \
    gs://gcp-pyspark-lakehouse-cdebatch52-486907/scripts/day2_01_bronze_probe.py \
    --cluster=pyspark-lakehouse-cluster \
    --region=us-central1 \
    -- \
    --bucket=gcp-pyspark-lakehouse-cdebatch52-486907

Key Day 1 learnings applied:
  - Read each month INDIVIDUALLY (never wildcard + mergeSchema)
  - spark.sql.parquet.enableVectorizedReader=false
  - spark.sql.parquet.mergeSchema=false
  - No Delta jar needed here (pure parquet read)
"""

import argparse
import json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, FloatType, IntegerType, LongType,
    StringType, TimestampType
)

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True, help="GCS bucket name (no gs:// prefix)")
args = parser.parse_args()

BUCKET       = args.bucket
BRONZE_PATH  = f"gs://{BUCKET}/bronze/nyc_taxi"
REPORT_PATH  = f"gs://{BUCKET}/logs/day2_bronze_probe_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"

# ── Spark Session (no Delta needed for probe) ──────────────────────────────────
spark = (
    SparkSession.builder
    .appName("Day2-BronzeProbe")
    .config("spark.sql.parquet.enableVectorizedReader", "false")
    .config("spark.sql.parquet.mergeSchema", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# ── Months to probe ────────────────────────────────────────────────────────────
MONTHS = [f"2023-{m:02d}" for m in range(1, 13)]

# Numeric columns we care about for range / outlier checks
NUMERIC_COLS = [
    "passenger_count",
    "trip_distance",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
]

TIMESTAMP_COLS = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]

# ── Probe ──────────────────────────────────────────────────────────────────────
report = {
    "probe_utc": datetime.utcnow().isoformat(),
    "bucket": BUCKET,
    "months": {},
    "schema_drift": [],
    "summary": {},
}

reference_schema = None  # first month becomes the baseline

for month in MONTHS:
    path = f"{BRONZE_PATH}/year=2023/month={month}/"
    print(f"\n{'='*60}")
    print(f"  Probing: {path}")
    print(f"{'='*60}")

    try:
        df = spark.read.parquet(path)
    except Exception as e:
        print(f"  [WARN] Could not read {month}: {e}")
        report["months"][month] = {"error": str(e)}
        continue

    month_report = {}

    # 1. Row count
    row_count = df.count()
    month_report["row_count"] = row_count
    print(f"  Rows : {row_count:,}")

    # 2. Schema
    schema_dict = {f.name: str(f.dataType) for f in df.schema.fields}
    month_report["schema"] = schema_dict
    print(f"  Columns ({len(schema_dict)}): {list(schema_dict.keys())}")

    # 3. Schema drift vs reference
    if reference_schema is None:
        reference_schema = schema_dict
        print("  [INFO] First month — set as reference schema.")
    else:
        added   = set(schema_dict) - set(reference_schema)
        removed = set(reference_schema) - set(schema_dict)
        retyped = {
            col: {"ref": reference_schema[col], "this": schema_dict[col]}
            for col in set(schema_dict) & set(reference_schema)
            if schema_dict[col] != reference_schema[col]
        }
        if added or removed or retyped:
            drift = {"month": month, "added": list(added),
                     "removed": list(removed), "retyped": retyped}
            report["schema_drift"].append(drift)
            print(f"  [DRIFT] added={added}  removed={removed}  retyped={retyped}")
        else:
            print("  Schema matches reference ✓")

    # 4. Null counts for all columns
    null_exprs = [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]
    null_row   = df.select(null_exprs).collect()[0].asDict()
    month_report["null_counts"] = null_row
    high_nulls = {k: v for k, v in null_row.items() if v and v > 0}
    if high_nulls:
        print(f"  Nulls: {high_nulls}")

    # 5. Numeric stats (min / max / mean / stddev) for key columns
    present_numeric = [c for c in NUMERIC_COLS if c in df.columns]
    if present_numeric:
        stat_exprs = []
        for c in present_numeric:
            col_cast = F.col(c).cast(DoubleType())
            stat_exprs += [
                F.min(col_cast).alias(f"{c}__min"),
                F.max(col_cast).alias(f"{c}__max"),
                F.mean(col_cast).alias(f"{c}__mean"),
                F.stddev(col_cast).alias(f"{c}__stddev"),
            ]
        stats_row = df.select(stat_exprs).collect()[0].asDict()
        # Round for readability
        stats_row = {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in stats_row.items()}
        month_report["numeric_stats"] = stats_row

        # Flag obvious outliers (fare / distance < 0)
        neg_fare = df.filter(F.col("fare_amount") < 0).count() if "fare_amount" in df.columns else 0
        neg_dist = df.filter(F.col("trip_distance") < 0).count() if "trip_distance" in df.columns else 0
        month_report["negative_fare_count"]     = neg_fare
        month_report["negative_distance_count"] = neg_dist
        if neg_fare or neg_dist:
            print(f"  [WARN] Negative fare={neg_fare}, negative distance={neg_dist}")

    # 6. Timestamp range
    present_ts = [c for c in TIMESTAMP_COLS if c in df.columns]
    if present_ts:
        ts_exprs = []
        for c in present_ts:
            ts_exprs += [
                F.min(c).cast(StringType()).alias(f"{c}__min"),
                F.max(c).cast(StringType()).alias(f"{c}__max"),
            ]
        ts_row = df.select(ts_exprs).collect()[0].asDict()
        month_report["timestamp_range"] = ts_row
        print(f"  Pickup range: {ts_row.get('tpep_pickup_datetime__min')} → "
              f"{ts_row.get('tpep_pickup_datetime__max')}")

    # 7. Distinct vendor IDs (quick sanity check)
    if "VendorID" in df.columns:
        vendors = [r[0] for r in df.select("VendorID").distinct().collect()]
        month_report["distinct_vendor_ids"] = vendors

    report["months"][month] = month_report
    df.unpersist()

# ── Summary across all months ──────────────────────────────────────────────────
total_rows   = sum(v.get("row_count", 0) for v in report["months"].values())
months_ok    = [m for m, v in report["months"].items() if "error" not in v]
months_err   = [m for m, v in report["months"].items() if "error" in v]

report["summary"] = {
    "total_rows":        total_rows,
    "months_ok":         months_ok,
    "months_error":      months_err,
    "schema_drift_count": len(report["schema_drift"]),
}

print(f"\n{'='*60}")
print(f"  PROBE COMPLETE")
print(f"  Total rows : {total_rows:,}")
print(f"  Months OK  : {months_ok}")
print(f"  Errors     : {months_err}")
print(f"  Schema drift events: {len(report['schema_drift'])}")
print(f"{'='*60}\n")

# ── Write JSON report to GCS ───────────────────────────────────────────────────
report_json = json.dumps(report, indent=2, default=str)

# Spark trick: write a single-partition text file to GCS
rdd = spark.sparkContext.parallelize([report_json], 1)
rdd.saveAsTextFile(REPORT_PATH)

print(f"Report written to: {REPORT_PATH}")

spark.stop()
