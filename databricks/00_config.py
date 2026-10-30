# Databricks notebook source
# MAGIC %md
# MAGIC # Config (run by every notebook with `%run ./00_config`)
# MAGIC Unity Catalog names, Volume paths, and the shared transform, model and monitoring code.

# COMMAND ----------

import os

CATALOG = "workspace"            # default catalog on Databricks Free Edition
SCHEMA = "drive_lakehouse"
NS = f"{CATALOG}.{SCHEMA}"
VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/raw"
ARCHIVE = f"{VOLUME}/archive"    # all 273 downloaded daily files
LANDING = f"{VOLUME}/landing"    # files "arrive" here; Auto Loader watches it
CHECKPOINTS = f"{VOLUME}/_checkpoints"

BRONZE = f"{NS}.bronze_smart"
SILVER = f"{NS}.silver_smart"
QUARANTINE = f"{NS}.silver_quarantine"
GOLD_FEATURES = f"{NS}.gold_drive_features"
GOLD_FLEET = f"{NS}.gold_fleet_daily"
RISK = f"{NS}.gold_drive_risk_daily"
MON_DQ = f"{NS}.monitor_dq_daily"
MON_SCORE = f"{NS}.monitor_score_daily"
MON_QUEUE = f"{NS}.monitor_queue_daily"
MON_RUNS = f"{NS}.monitor_run_log"
MODEL_NAME = f"{NS}.drive_failure_7d"
EXPERIMENT = "/Users/amarnathreddyganta5@gmail.com/drive-lakehouse/drive_failure_7d"

SOURCE_COMMIT = "d7e8e5968a5a01b23fdf58864ddef1a42ffe97dc"
SOURCE_URL = ("https://raw.githubusercontent.com/Phillweston/Disk-Failure-Prediction/"
              f"{SOURCE_COMMIT}/data_preprocess/{{day}}.csv")
print("namespace:", NS)

# COMMAND ----------

# MAGIC %run ./lakehouse_transforms

# COMMAND ----------

# MAGIC %run ./lakehouse_model

# COMMAND ----------

# MAGIC %run ./lakehouse_monitoring

# COMMAND ----------

# MAGIC %run ./raw_manifest

# COMMAND ----------

# MAGIC %run ./reference_metrics
