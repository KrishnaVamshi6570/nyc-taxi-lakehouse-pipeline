# 🚕 NYC Taxi Lakehouse Pipeline — GCP + PySpark + Delta Lake

A production-grade, end-to-end **medallion lakehouse architecture** built on Google Cloud Platform using PySpark, Delta Lake, BigQuery, Cloud Composer (Airflow), and Looker Studio — powered by the NYC Yellow Taxi 2023 dataset (38.3M rows).

---

## 📐 Architecture
```
Raw Parquet (GCS)        Delta Lake (GCS)         BigQuery + Looker Studio
┌─────────────┐         ┌─────────────┐           ┌──────────────────────┐
│   BRONZE    │──────▶  │   SILVER    │──────────▶│        GOLD          │
│  38.3M rows │  clean  │  35.6M rows │  aggregate│  gold_hourly_stats   │
│  per-month  │  derive │  Delta fmt  │           │  gold_location_stats │
│  parquet    │         │  features   │           │  gold_monthly_stats  │
└─────────────┘         └─────────────┘           └──────────────────────┘
GCS Bucket                                        BQ Dataset
**GCS Bucket:** `gs://gcp-pyspark-lakehouse-cdebatch52-486907`
**BigQuery Dataset:** `cdebatch52-486907.lakehouse_gold`
**Orchestration:** Cloud Composer 2 (Airflow 2.10) — 8-task DAG (~33 min end-to-end)
```
---


## 📁 Repository Structure
```
nyc-taxi-lakehouse-pipeline/
├── scripts/                           # PySpark jobs (deployed to GCS)
│   ├── 01_validate_bronze.py          # Bronze validation
│   ├── day2_01_bronze_probe.py        # Bronze ingestion probe
│   ├── day2_02_silver_transform.py    # Silver: cleaning + feature engineering
│   ├── day3_03_gold_transform.py      # Gold: hourly + location aggregations
│   ├── day3_03b_gold_monthly.py       # Gold: monthly aggregations (year-filtered)
│   ├── day3_04_bq_load.py             # BigQuery loader (all 3 gold tables)
│   ├── day3_probe_silver.py           # Silver probe/validation
│   ├── day4_05_ml_tip_prediction.py   # MLlib GBT tip prediction model
│   ├── day4_probe_silver.py           # Silver probe Day 4
│   └── check_model.py                 # Model validation script
├── dags/
│   └── lakehouse_pipeline_dag.py      # Airflow DAG (8 tasks, end-to-end)
├── .gitignore
└── README.md
```
---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Compute | Google Dataproc (PySpark 3.x, e2-standard-4) |
| Storage | Google Cloud Storage (GCS) |
| Table Format | Delta Lake 2.3.0 |
| Data Warehouse | BigQuery (`lakehouse_gold` dataset) |
| Orchestration | Cloud Composer 2 / Apache Airflow 2.10 |
| ML | PySpark MLlib (Gradient Boosted Trees) |
| Dashboard | Looker Studio (connected to BigQuery) |
| Language | Python 3, PySpark SQL |

---

## 📅 Project Timeline (7 Days)

| Day | Focus | Key Output |
|-----|-------|-----------|
| 1 | Bronze Ingestion | 38.3M rows loaded from NYC TLC parquet files |
| 2 | Silver Delta Lake | 35.6M cleaned rows + derived features (tip_pct, speed_mph, etc.) |
| 3 | Gold + BigQuery | 3 aggregation tables loaded into BQ |
| 4 | ML Model | GBT tip prediction: RMSE=9.06, R²=0.57 |
| 5 | Airflow DAG | 8-task pipeline orchestrated end-to-end in ~33 min |
| 6 | Dashboard + Hardening | Looker Studio live, ML fixed, region migrated to us-east1 |
| 7 | Fixes + GitHub | SMTP fix, bad-year data filter, BQ reload, repo published |

---

## ⚙️ Critical Spark Config

```python
spark = SparkSession.builder \
    .config("spark.jars.packages", "io.delta:delta-core_2.12:2.3.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.sql.parquet.enableVectorizedReader", "false") \
    .config("spark.sql.parquet.mergeSchema", "false") \
    .getOrCreate()
```

> ⚠️ `enableVectorizedReader=false` is mandatory — NYC TLC dataset has mixed INT32/INT64 timestamp encoding that causes silent data corruption otherwise.

---

## 🥇 Gold Tables (BigQuery)

| Table | Rows | Granularity |
|-------|------|-------------|
| `gold_hourly_stats` | 168 | pickup_hour + day_of_week |
| `gold_location_stats` | 262 | pickup zone ID |
| `gold_monthly_stats` | 15 | year + month (filtered >= 2020) |

---

## 🤖 ML Model — Tip Prediction

- **Algorithm:** Gradient Boosted Trees (GBTRegressor)
- **Sample:** 500K rows from Silver
- **Results:** RMSE = 9.06 | R² = 0.57
- **Saved to:** `gs://<bucket>/models/gbt_tip_model`

---

## 🔄 Airflow DAG — `lakehouse_pipeline`
```
bronze_ingest → silver_transform → gold_hourly  ──┐
gold_monthly  ──┼── bq_load → ml_tip_prediction → complete
gold_location ──┘
```
8 tasks, ~33 minutes total runtime.

---

## 🚀 How to Reproduce

### Prerequisites
- GCP project with Dataproc, BigQuery, GCS, Composer APIs enabled
- GCS bucket + BigQuery dataset `lakehouse_gold` created

### Step 1 — Upload scripts to GCS
```bash
gsutil cp scripts/* gs://<your-bucket>/scripts/
gsutil cp dags/lakehouse_pipeline_dag.py gs://<your-composer-dags-bucket>/dags/
```

### Step 2 — Create Dataproc cluster
```bash
gcloud dataproc clusters create pyspark-lakehouse-cluster \
  --region=us-east1 \
  --subnet=projects/<project>/regions/us-east1/subnetworks/default \
  --master-machine-type=e2-standard-4 \
  --num-workers=2 --worker-machine-type=e2-standard-4 \
  --image-version=2.1-debian11 \
  --max-idle=1h
```

### Step 3 — Trigger DAG
In Airflow UI: **DAGs → lakehouse_pipeline → Trigger DAG**

---

## 📝 Known Issues & Fixes Applied

| Issue | Fix |
|-------|-----|
| NYC TLC mixed INT32/INT64 parquet timestamps | `enableVectorizedReader=false` |
| Corrupt Silver rows with years 2001/2003/2009 | Filter `pickup_year >= 2020` in gold monthly |
| `us-central1` quota exhaustion | Migrated Dataproc to `us-east1` |
| Composer SMTP using SendGrid backend | Override `email-email_backend` airflow config |
| ML script retraining every DAG run (15+ hrs) | Fixed to load pre-trained model |
| Default VPC network not found on cluster create | Use explicit `--subnet` flag |

---
---

## 📊 Dashboard Preview



*Looker Studio dashboard connected to BigQuery gold tables — avg fare by hour, trip count by location, monthly stats table.*

---

## 👤 Author

**Kurma Vamshi** — Data Engineering 7-Day GCP Lakehouse Project
