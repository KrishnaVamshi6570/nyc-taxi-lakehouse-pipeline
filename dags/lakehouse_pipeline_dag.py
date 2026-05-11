"""
lakehouse_pipeline_dag.py
GCP PySpark Lakehouse Pipeline — Full Orchestration DAG
Day 6 version: email_on_failure, retries, quota-safe cluster (n1-standard-2 x3 VMs).
Schedule: Monthly on the 1st at 06:00 UTC (matching NYC TLC data cadence)
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import (
    DataprocCreateClusterOperator,
    DataprocDeleteClusterOperator,
    DataprocSubmitJobOperator,
)

# ── Project-level constants ────────────────────────────────────────────────────
PROJECT_ID      = "cdebatch52-486907"
REGION          = "us-east1"
CLUSTER_NAME    = "pyspark-lakehouse-cluster"
GCS_BUCKET      = "gcp-pyspark-lakehouse-cdebatch52-486907"
SCRIPTS_BASE    = f"gs://{GCS_BUCKET}/scripts"
BQ_TEMP_BUCKET  = GCS_BUCKET

# ── Spark properties applied to every PySpark job ─────────────────────────────
SPARK_PROPERTIES = {
    "spark.jars.packages":                        "io.delta:delta-core_2.12:2.3.0",
    "spark.sql.extensions":                       "io.delta.sql.DeltaSparkSessionExtension",
    "spark.sql.catalog.spark_catalog":            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
    "spark.sql.parquet.enableVectorizedReader":   "false",
    "spark.sql.parquet.mergeSchema":              "false",
}

# Delta configs not needed for bronze ingest (no Delta read/write)
SPARK_PROPERTIES_BRONZE = {
    "spark.sql.parquet.enableVectorizedReader":   "false",
    "spark.sql.parquet.mergeSchema":              "false",
}

# BQ connector loaded via --jars (NOT spark.jars.packages — comma in Maven
# coord breaks gcloud arg parsing, as learned on Day 3)
BQ_JAR = "gs://spark-lib/bigquery/spark-bigquery-with-dependencies_2.12-0.32.2.jar"

# ── Cluster definition ────────────────────────────────────────────────────────
# Quota-safe: Composer env consumes most of the project quota.
# Available: 12 CPUs, 3 IPs. This cluster uses 6 CPUs and 3 IPs exactly.
# n1-standard-2 = 2 vCPUs: 1 master + 2 workers = 6 CPUs, 3 VMs.
# Spot secondary workers removed until quota is increased.
CLUSTER_CONFIG = {
    "master_config": {
        "num_instances": 1,
        "machine_type_uri": "n1-standard-2",
        "disk_config": {"boot_disk_size_gb": 100},
    },
    "worker_config": {
        "num_instances": 2,
        "machine_type_uri": "n1-standard-2",
        "disk_config": {"boot_disk_size_gb": 100},
    },
    "software_config": {
        "image_version": "2.1-debian11",
        "optional_components": ["JUPYTER"],
    },
    "lifecycle_config": {
        "idle_delete_ttl": {"seconds": 3600},
    },
    "endpoint_config": {"enable_http_port_access": True},
}

# ── Default args ───────────────────────────────────────────────────────────────
default_args = {
    "owner":             "airflow",
    "depends_on_past":   False,
    "email":             ["vamshiqtsn@gmail.com"],
    "email_on_failure":  True,
    "email_on_retry":    False,
    "retries":           1,
    "retry_delay":       timedelta(minutes=5),
    "start_date":        datetime(2026, 5, 1),
}

# ── DAG ───────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="lakehouse_pipeline",
    default_args=default_args,
    description="NYC Taxi Lakehouse — Bronze → Silver → Gold → BQ → ML (monthly)",
    schedule_interval="0 6 1 * *",
    catchup=False,
    tags=["lakehouse", "pyspark", "delta", "nyc-taxi"],
) as dag:

    # ── Task 1: Create Dataproc cluster ───────────────────────────────────────
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_cluster",
        project_id=PROJECT_ID,
        cluster_name=CLUSTER_NAME,
        cluster_config=CLUSTER_CONFIG,
        region=REGION,
    )

    # ── Task 2: Bronze ingestion ──────────────────────────────────────────────
    # Script name verified on Day 5: day2_01_bronze_probe.py (NOT 01_ingest_bronze.py)
    ingest_bronze = DataprocSubmitJobOperator(
        task_id="ingest_bronze",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {"cluster_name": CLUSTER_NAME},
            "pyspark_job": {
                "main_python_file_uri": f"{SCRIPTS_BASE}/day2_01_bronze_probe.py",
                "properties": SPARK_PROPERTIES_BRONZE,
                "args": [f"--bucket={GCS_BUCKET}"],
            },
        },
    )

    # ── Task 3: Silver transform ──────────────────────────────────────────────
    silver_transform = DataprocSubmitJobOperator(
        task_id="silver_transform",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {"cluster_name": CLUSTER_NAME},
            "pyspark_job": {
                "main_python_file_uri": f"{SCRIPTS_BASE}/day2_02_silver_transform.py",
                "properties": SPARK_PROPERTIES,
                "args": [f"--bucket={GCS_BUCKET}"],
            },
        },
    )

    # ── Task 4: Gold hourly + location aggregations ───────────────────────────
    gold_hourly_location = DataprocSubmitJobOperator(
        task_id="gold_hourly_location",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {"cluster_name": CLUSTER_NAME},
            "pyspark_job": {
                "main_python_file_uri": f"{SCRIPTS_BASE}/day3_03_gold_transform.py",
                "properties": SPARK_PROPERTIES,
                "args": [f"--bucket={GCS_BUCKET}"],
            },
        },
    )

    # ── Task 5: Gold monthly aggregation ─────────────────────────────────────
    gold_monthly = DataprocSubmitJobOperator(
        task_id="gold_monthly",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {"cluster_name": CLUSTER_NAME},
            "pyspark_job": {
                "main_python_file_uri": f"{SCRIPTS_BASE}/day3_03b_gold_monthly.py",
                "properties": SPARK_PROPERTIES,
                "args": [f"--bucket={GCS_BUCKET}"],
            },
        },
    )

    # ── Task 6: BigQuery load ─────────────────────────────────────────────────
    bq_load = DataprocSubmitJobOperator(
        task_id="bq_load",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {"cluster_name": CLUSTER_NAME},
            "pyspark_job": {
                "main_python_file_uri": f"{SCRIPTS_BASE}/day3_04_bq_load.py",
                "properties": SPARK_PROPERTIES,
                "jar_file_uris": [BQ_JAR],
                "args": [f"--bucket={GCS_BUCKET}"],
            },
        },
    )

    # ── Task 7: ML scoring ────────────────────────────────────────────────────
    ml_score = DataprocSubmitJobOperator(
        task_id="ml_score",
        project_id=PROJECT_ID,
        region=REGION,
        job={
            "placement": {"cluster_name": CLUSTER_NAME},
            "pyspark_job": {
                "main_python_file_uri": f"{SCRIPTS_BASE}/day4_05_ml_tip_prediction.py",
                "properties": SPARK_PROPERTIES,
                "args": [f"--bucket={GCS_BUCKET}"],
            },
        },
    )

    # ── Task 8: Delete cluster ────────────────────────────────────────────────
    # trigger_rule="all_done" ensures cleanup even if upstream tasks fail
    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_cluster",
        project_id=PROJECT_ID,
        cluster_name=CLUSTER_NAME,
        region=REGION,
        trigger_rule="all_done",
    )

    # ── Task chain ────────────────────────────────────────────────────────────
    (
        create_cluster
        >> ingest_bronze
        >> silver_transform
        >> gold_hourly_location
        >> gold_monthly
        >> bq_load
        >> ml_score
        >> delete_cluster
    )
