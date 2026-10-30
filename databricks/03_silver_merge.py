# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Silver: validate, quarantine, and MERGE (streaming foreachBatch)
# MAGIC Reads new bronze rows as a stream, applies the shared `bronze_to_silver` rules, appends
# MAGIC failing rows to a quarantine table, and MERGEs clean rows into silver on
# MAGIC `(serial_number, date)`. Matched rows update only when their content hash changed, so
# MAGIC replaying a file is a no-op. `maintenance = yes` also runs the replay check, OPTIMIZE /
# MAGIC Z-ORDER, history, and time travel.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
from pyspark.sql import functions as F

dbutils.widgets.dropdown("maintenance", "no", ["no", "yes"])
maintenance = dbutils.widgets.get("maintenance") == "yes"

# Create empty target tables with the exact transform output schema.
empty_silver, empty_quar = bronze_to_silver(spark.table(BRONZE).limit(0))
if not spark.catalog.tableExists(SILVER):
    empty_silver.write.saveAsTable(SILVER)
if not spark.catalog.tableExists(QUARANTINE):
    empty_quar.write.saveAsTable(QUARANTINE)

MERGE_SQL = f"""
MERGE INTO {SILVER} t
USING silver_updates s
  ON t.serial_number = s.serial_number AND t.date = s.date
WHEN MATCHED AND t._row_hash <> s._row_hash THEN UPDATE SET *
WHEN NOT MATCHED THEN INSERT *
"""


def upsert(batch_df, batch_id):
    silver, quarantine = bronze_to_silver(batch_df)
    quarantine.write.mode("append").saveAsTable(QUARANTINE)
    silver.createOrReplaceTempView("silver_updates")
    batch_df.sparkSession.sql(MERGE_SQL)


q = (spark.readStream.table(BRONZE).writeStream
     .foreachBatch(upsert)
     .option("checkpointLocation", f"{CHECKPOINTS}/silver")
     .trigger(availableNow=True)
     .start())
q.awaitTermination()

# COMMAND ----------

s = spark.sql(f"""SELECT count(*) AS n, count(DISTINCT serial_number) AS drives,
                         sum(failure) AS failures FROM {SILVER}""").first()
quar = spark.sql(f"SELECT count(*) AS n FROM {QUARANTINE}").first().n
out = {"silver_rows": s.n, "drives": s.drives, "failures": s.failures, "quarantine_rows": quar}

# COMMAND ----------

if maintenance:
    REPLAY_DAY = "2018-09-15"
    replay = spark.table(BRONZE).filter(F.col("_source_file").endswith(f"{REPLAY_DAY}.csv"))
    silver_replay, _ = bronze_to_silver(replay)
    silver_replay.createOrReplaceTempView("silver_updates")
    spark.sql(MERGE_SQL)
    h = spark.sql(f"DESCRIBE HISTORY {SILVER} LIMIT 1").first()
    out["replay"] = {"day": REPLAY_DAY, "operation": h.operation,
                     "source_rows": h.operationMetrics.get("numSourceRows"),
                     "inserted": h.operationMetrics.get("numTargetRowsInserted"),
                     "updated": h.operationMetrics.get("numTargetRowsUpdated")}

    files_before = spark.sql(f"DESCRIBE DETAIL {SILVER}").first().numFiles
    spark.sql(f"OPTIMIZE {SILVER} ZORDER BY (serial_number)")
    files_after = spark.sql(f"DESCRIBE DETAIL {SILVER}").first().numFiles
    out["optimize"] = {"files_before": files_before, "files_after": files_after}

    hist = spark.sql(f"DESCRIBE HISTORY {SILVER}")
    first_merge = hist.filter("operation = 'MERGE'").orderBy("version").first().version
    out["history"] = [(r.version, r.operation) for r in hist.orderBy("version").collect()]
    out["time_travel"] = {
        "first_merge_version": first_merge,
        "rows_at_first_merge": spark.sql(f"SELECT count(*) AS n FROM {SILVER} VERSION AS OF {first_merge}").first().n,
        "rows_now": spark.table(SILVER).count(),
    }

print(json.dumps(out, default=str, indent=2))
dbutils.notebook.exit(json.dumps(out, default=str))
