# Databricks notebook source
# MAGIC %md
# MAGIC # 02 Bronze: incremental file ingestion with Auto Loader
# MAGIC Copies a batch of files from `archive/` into `landing/` (simulating daily drops), then
# MAGIC Auto Loader ingests only files it has not seen before (`trigger(availableNow=True)`).
# MAGIC Every column stays a string in bronze; lineage columns record the source file and load time.
# MAGIC
# MAGIC `batch`: `initial` lands Jan to Aug, `next_day` lands one more day (the daily job),
# MAGIC `all_remaining` lands everything left. Reruns ingest nothing new: the checkpoint tracks files.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

dbutils.widgets.dropdown("batch", "next_day", ["initial", "next_day", "all_remaining"])
batch = dbutils.widgets.get("batch")

archive = sorted(f for f in os.listdir(ARCHIVE) if f.endswith(".csv"))
landed = set(os.listdir(LANDING))
pending = [f for f in archive if f not in landed]
if batch == "initial":
    to_land = [f for f in pending if f < "2018-09-01.csv"]
elif batch == "next_day":
    to_land = pending[:1]
else:
    to_land = pending
for f in to_land:
    dbutils.fs.cp(f"{ARCHIVE}/{f}", f"{LANDING}/{f}")
print(f"landed {len(to_land)} file(s): {to_land[:1]} .. {to_land[-1:]}")

# COMMAND ----------

from pyspark.sql import types as T

schema = T.StructType([T.StructField(c, T.StringType()) for c in RAW_COLS])
stream = (spark.readStream.format("cloudFiles")
          .option("cloudFiles.format", "csv")
          .option("header", "true")
          .schema(schema)
          .load(LANDING))
q = (add_bronze_metadata(stream).writeStream
     .option("checkpointLocation", f"{CHECKPOINTS}/bronze")
     .trigger(availableNow=True)
     .toTable(BRONZE))
q.awaitTermination()

# COMMAND ----------

r = spark.sql(f"SELECT count(*) AS n, count(DISTINCT _source_file) AS files FROM {BRONZE}").first()
print(f"bronze rows={r.n} files={r.files}")
dbutils.notebook.exit(f"batch={batch} landed={len(to_land)} bronze_rows={r.n} bronze_files={r.files}")
