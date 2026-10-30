# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Setup: schema, Volume, and raw data download
# MAGIC Downloads the 273 daily Backblaze SMART snapshots (ST4000DM000, 2018-01-01 to 2018-09-30)
# MAGIC into a Unity Catalog Volume and checks a combined SHA-256 digest against the local manifest (`raw_manifest` notebook).
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

import hashlib
import urllib.request

downloaded = 0
for name in MANIFEST_FILES:
    dest = os.path.join(ARCHIVE, name)
    if os.path.exists(dest):
        continue
    with urllib.request.urlopen(SOURCE_URL.format(day=name[:-4]), timeout=120) as r:
        data = r.read()
    with open(dest, "wb") as out:
        out.write(data)
    downloaded += 1
print("downloaded", downloaded, "files")

# COMMAND ----------

hashes = [hashlib.sha256(open(os.path.join(ARCHIVE, n), "rb").read()).hexdigest() for n in MANIFEST_FILES]
digest = hashlib.sha256("".join(hashes).encode()).hexdigest()
print(f"{len(hashes)} files hashed; combined digest matches reference: {digest == MANIFEST_DIGEST}")
assert digest == MANIFEST_DIGEST
dbutils.notebook.exit(f"files={len(hashes)} digest_ok={digest == MANIFEST_DIGEST} downloaded={downloaded}")
