from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
import pyspark.sql.functions as F

PROJECT_ID = "cdebatch52-486907"
BUCKET = f"gs://gcp-pyspark-lakehouse-{PROJECT_ID}"

spark = (
    SparkSession.builder
    .appName("validate-bronze")
    .config("spark.sql.parquet.enableVectorizedReader", "false")
    .config("spark.sql.parquet.mergeSchema", "false")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

MONTHS = [f"2023-{str(m).zfill(2)}" for m in range(1, 13)]

CAST_COLS = {
    "VendorID":              LongType(),
    "tpep_pickup_datetime":  TimestampType(),
    "tpep_dropoff_datetime": TimestampType(),
    "passenger_count":       DoubleType(),
    "trip_distance":         DoubleType(),
    "RatecodeID":            DoubleType(),
    "store_and_fwd_flag":    StringType(),
    "PULocationID":          LongType(),
    "DOLocationID":          LongType(),
    "payment_type":          LongType(),
    "fare_amount":           DoubleType(),
    "extra":                 DoubleType(),
    "mta_tax":               DoubleType(),
    "tip_amount":            DoubleType(),
    "tolls_amount":          DoubleType(),
    "improvement_surcharge": DoubleType(),
    "total_amount":          DoubleType(),
    "congestion_surcharge":  DoubleType(),
    "Airport_fee":           DoubleType(),
}

def read_month(month: str) -> DataFrame:
    """Read a single month file with its own inferred schema, then cast."""
    year = month[:4]
    mon  = month[5:]
    path = f"{BUCKET}/bronze/nyc_taxi/year={year}/month={mon}/yellow_tripdata_{month}.parquet"
    df = spark.read \
        .option("mergeSchema", "false") \
        .parquet(path)
    # Cast each column to canonical type; add as null if missing in this file
    for col_name, col_type in CAST_COLS.items():
        if col_name in df.columns:
            df = df.withColumn(col_name, F.col(col_name).cast(col_type))
        else:
            df = df.withColumn(col_name, F.lit(None).cast(col_type))
    return df.select(list(CAST_COLS.keys()))

# Build union of all months
print("Reading months individually...")
dfs = []
for month in MONTHS:
    print(f"  Loading {month}...")
    dfs.append(read_month(month))

df = dfs[0]
for d in dfs[1:]:
    df = df.unionAll(d)

print("\n" + "="*50)
print(f"Bronze row count : {df.count():,}")
print("Bronze schema    :")
df.printSchema()

print("\nDate range:")
df.select(
    F.min("tpep_pickup_datetime"),
    F.max("tpep_pickup_datetime")
).show()

print("\nNull counts per column:")
df.select([
    F.count(F.when(F.col(c).isNull(), c)).alias(c)
    for c in CAST_COLS.keys()
]).show(vertical=True)

print("\nSample rows:")
df.show(5, truncate=False)
print("="*50)

spark.stop()
