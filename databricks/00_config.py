# Databricks notebook source
# MAGIC %md
# MAGIC # Config (run by every notebook with `%run ./00_config`)
# MAGIC Unity Catalog names, Volume paths, and the shared `lakehouse` package from this Git folder.

# COMMAND ----------

import os
import sys

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
MODEL_NAME = f"{NS}.drive_failure_7d"

# Make the repo root importable (this notebook lives in <repo>/databricks/).
REPO_ROOT = os.path.dirname(os.getcwd())
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SOURCE_COMMIT = "d7e8e5968a5a01b23fdf58864ddef1a42ffe97dc"
SOURCE_URL = ("https://raw.githubusercontent.com/Phillweston/Disk-Failure-Prediction/"
              f"{SOURCE_COMMIT}/data_preprocess/{{day}}.csv")
print("namespace:", NS, "| repo root:", REPO_ROOT)
