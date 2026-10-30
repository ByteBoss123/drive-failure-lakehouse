# Databricks notebook source
# MAGIC %md
# MAGIC # 03 Silver: validate, quarantine, and MERGE (streaming foreachBatch)
# MAGIC Reads new bronze rows as a stream, applies the shared `bronze_to_silver` rules, appends
# MAGIC failing rows to a quarantine table, and MERGEs clean rows into silver on
# MAGIC `(serial_number, date)`. Matched rows update only when their content hash changed, so
# MAGIC replaying a file is a no-op. Then OPTIMIZE / Z-ORDER, history, and time travel.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

from pyspark.sql import functions as F
from lakehouse.transforms import bronze_to_silver

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

display(spark.sql(f"""SELECT count(*) AS silver_rows, count(DISTINCT serial_number) AS drives,
                             sum(failure) AS failures, min(date) AS first_day, max(date) AS last_day
                      FROM {SILVER}"""))
display(spark.sql(f"SELECT dq_reason, count(*) AS rows FROM {QUARANTINE} GROUP BY dq_reason"))

# COMMAND ----------

# MAGIC %md ## Idempotency check: replay one September day straight from bronze

# COMMAND ----------

REPLAY_DAY = "2018-09-15"
replay = spark.table(BRONZE).filter(F.col("_source_file").endswith(f"{REPLAY_DAY}.csv"))
if replay.limit(1).count():
    silver_replay, _ = bronze_to_silver(replay)
    silver_replay.createOrReplaceTempView("silver_updates")
    spark.sql(MERGE_SQL)
    display(spark.sql(f"DESCRIBE HISTORY {SILVER} LIMIT 1")
            .select("version", "operation", "operationMetrics.numSourceRows",
                    "operationMetrics.numTargetRowsInserted", "operationMetrics.numTargetRowsUpdated"))
else:
    print(f"{REPLAY_DAY} not ingested yet; run 02 with next_day/all_remaining first")

# COMMAND ----------

# MAGIC %md ## OPTIMIZE + Z-ORDER, history, time travel

# COMMAND ----------

display(spark.sql(f"OPTIMIZE {SILVER} ZORDER BY (serial_number)"))

# COMMAND ----------

hist = spark.sql(f"DESCRIBE HISTORY {SILVER}").select("version", "timestamp", "operation", "operationMetrics")
display(hist)

# COMMAND ----------

first_merge = (hist.filter("operation = 'MERGE'").agg(F.min("version")).first()[0])
display(spark.sql(f"""
  SELECT (SELECT count(*) FROM {SILVER} VERSION AS OF {first_merge}) AS rows_after_first_merge,
         (SELECT count(*) FROM {SILVER}) AS rows_now"""))
