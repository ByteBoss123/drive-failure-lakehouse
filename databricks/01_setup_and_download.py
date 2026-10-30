# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Setup: schema, Volume, and raw data download
# MAGIC Downloads the 273 daily Backblaze SMART snapshots (ST4000DM000, 2018-01-01 to 2018-09-30)
# MAGIC into a Unity Catalog Volume and checks each file's SHA-256 against `data/raw_manifest.csv`.
# MAGIC
# MAGIC If the download cell fails because outbound internet is blocked, upload the CSVs to
# MAGIC `/Volumes/workspace/drive_lakehouse/raw/archive/` through Catalog Explorer and rerun the
# MAGIC checksum cell.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {NS} COMMENT 'Backblaze drive SMART telemetry lakehouse'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {NS}.raw COMMENT 'Raw daily CSV drops and stream checkpoints'")
for p in (ARCHIVE, LANDING, CHECKPOINTS):
    os.makedirs(p, exist_ok=True)

# COMMAND ----------

import csv
import hashlib
import urllib.request

with open(os.path.join(REPO_ROOT, "data", "raw_manifest.csv")) as f:
    manifest = {r["file"]: r["sha256"] for r in csv.DictReader(f)}

downloaded = 0
for name in sorted(manifest):
    dest = os.path.join(ARCHIVE, name)
    if os.path.exists(dest):
        continue
    with urllib.request.urlopen(SOURCE_URL.format(day=name[:-4]), timeout=120) as r, open(dest, "wb") as out:
        out.write(r.read())
    downloaded += 1
print("downloaded", downloaded, "files")

# COMMAND ----------

bad = [n for n in sorted(manifest)
       if hashlib.sha256(open(os.path.join(ARCHIVE, n), "rb").read()).hexdigest() != manifest[n]]
present = len([n for n in manifest if os.path.exists(os.path.join(ARCHIVE, n))])
print(f"{present}/{len(manifest)} files present, checksum mismatches: {len(bad)}")
assert present == len(manifest) and not bad, bad[:5]
