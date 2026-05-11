"""Day 6 revision v6 - provide both column names"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--bucket", required=True)
args = parser.parse_args()
BUCKET = args.bucket

SILVER_PATH  = f"gs://{BUCKET}/silver/nyc_taxi_delta"
GOLD_HOURLY  = f"gs://{BUCKET}/gold/gold_hourly_stats"
GOLD_LOC     = f"gs://{BUCKET}/gold/gold_location_stats"
MODEL_PATH   = f"gs://{BUCKET}/models/tip_prediction_gbt"
OUTPUT_PATH  = f"gs://{BUCKET}/models/tip_prediction_scores"
METRICS_PATH = f"gs://{BUCKET}/models/tip_prediction_gbt_metrics"
TARGET_YEAR  = 2023
SAMPLE_ROWS  = 500000

spark = (SparkSession.builder.appName("day4_tip_scoring_v6")
    .config("spark.sql.shuffle.partitions","100").getOrCreate())
spark.sparkContext.setLogLevel("WARN")

model = PipelineModel.load(MODEL_PATH)
expected = model.stages[0].getInputCols()
print(f"Model expects these {len(expected)} features: {expected}")

silver_raw = spark.read.format("delta").load(SILVER_PATH)
silver_raw = silver_raw.withColumn("pickup_year", F.year("pickup_datetime"))
silver_2023 = silver_raw.filter(F.col("pickup_year") == TARGET_YEAR)
total = silver_2023.count()
fraction = min(1.0, SAMPLE_ROWS / max(total,1))
sample = silver_2023.sample(fraction=fraction, seed=42)
print(f"Sample: {int(total*fraction):,} rows")

gold_hourly = spark.read.format("delta").load(GOLD_HOURLY).select(
    F.col("pickup_hour"), F.col("pickup_dayofweek"),
    F.col("avg_fare").alias("g_avg_fare_hour"),
    F.col("avg_tip_pct").alias("g_avg_tip_pct_hour"),
    F.col("avg_trip_duration_minutes").alias("g_avg_duration_hour"),
    F.col("avg_speed_mph").alias("g_avg_speed_hour"))

# Provide BOTH names — model will use whichever it needs
gold_loc = spark.read.format("delta").load(GOLD_LOC).select(
    F.col("pu_location_id"),
    F.col("avg_fare").alias("g_avg_fare_loc"),
    F.col("avg_tip_pct").alias("g_avg_tip_pct_loc"),
    F.col("avg_trip_duration_minutes").alias("g_avg_duration_loc"),
    F.col("airport_trip_rate").alias("g_airport_rate_loc"),
    F.col("airport_trip_rate").alias("is_airport_trip"))

base = sample.select(
    F.col("tip_pct").cast(DoubleType()),
    F.col("trip_distance").cast(DoubleType()),
    F.col("fare_amount").cast(DoubleType()),
    F.col("trip_duration_minutes").cast(DoubleType()),
    F.col("speed_mph").cast(DoubleType()),
    F.col("passenger_count").cast(DoubleType()),
    F.col("pickup_hour").cast(DoubleType()),
    F.col("pickup_dayofweek").cast(DoubleType()),
    F.col("pu_location_id").cast(DoubleType()),
    F.col("do_location_id").cast(DoubleType()),
    F.col("payment_type").cast(DoubleType()),
    F.col("ratecode_id").cast(DoubleType()),
    F.col("vendor_id").cast(DoubleType()))

enriched = (base
    .join(gold_hourly,
          (base.pickup_hour==gold_hourly.pickup_hour)&
          (base.pickup_dayofweek==gold_hourly.pickup_dayofweek),"left")
    .join(gold_loc, base.pu_location_id==gold_loc.pu_location_id,"left")
    .drop(gold_hourly.pickup_hour).drop(gold_hourly.pickup_dayofweek)
    .drop(gold_loc.pu_location_id))

print(f"Enriched cols: {sorted(enriched.columns)}")

# Check all expected features exist
missing = [f for f in expected if f not in enriched.columns]
if missing:
    raise ValueError(f"Still missing: {missing}. Have: {sorted(enriched.columns)}")

predictions = model.transform(enriched)
predictions.cache()
pred_count = predictions.count()
print(f"Scored {pred_count:,} rows")

rmse = RegressionEvaluator(labelCol="tip_pct",predictionCol="prediction",metricName="rmse").evaluate(predictions)
r2   = RegressionEvaluator(labelCol="tip_pct",predictionCol="prediction",metricName="r2").evaluate(predictions)
mae  = RegressionEvaluator(labelCol="tip_pct",predictionCol="prediction",metricName="mae").evaluate(predictions)

predictions.select("tip_pct","prediction").write.mode("overwrite").parquet(OUTPUT_PATH)
spark.createDataFrame([("RMSE",float(rmse)),("MAE",float(mae)),("R2",float(r2)),("scored_rows",float(pred_count))],
    ["metric","value"]).write.mode("overwrite").parquet(METRICS_PATH)
print(f"COMPLETE | RMSE={rmse:.4f} | R2={r2:.4f} | Scored={pred_count:,}")
spark.stop()
