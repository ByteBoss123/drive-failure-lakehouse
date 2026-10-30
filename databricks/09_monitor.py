# Databricks notebook source
# MAGIC %md
# MAGIC # 09 Monitor: data quality, score drift, and queue performance
# MAGIC * `monitor_dq_daily`: rows kept and quarantined per data day, sensor faults, DQ alert.
# MAGIC * `monitor_score_daily`: score distribution per day and PSI against the first 7 scored days.
# MAGIC * `monitor_queue_daily`: once a day's 7-day outcome is known, how many of the queued drives
# MAGIC   actually failed, against an equal-size queue ranked by the SMART error counters.
# MAGIC * `monitor_run_log`: one appended row per run with headline numbers and alert counts.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import datetime as dt
import json
from pyspark.sql import functions as F

silver = spark.table(SILVER)
gold = spark.table(GOLD_FEATURES)
risk = spark.table(RISK)

dq = dq_daily(silver, spark.table(QUARANTINE), _parse_date)
dq.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(MON_DQ)

drift = score_drift_daily(risk)
spark.createDataFrame(drift).withColumn("score_date", F.col("score_date").cast("date")) \
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(MON_SCORE)

latest_data = gold.agg(F.max("date")).first()[0]
last_labeled = latest_data - dt.timedelta(days=LABEL_HORIZON_DAYS - 1)
perf = queue_performance_daily(risk, gold, last_labeled)
perf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(MON_QUEUE)

# COMMAND ----------

p = perf.agg(F.sum("queue_size").alias("q"), F.sum("model_hits").alias("m"),
             F.sum("rule_hits").alias("r"), F.count("*").alias("days")).first()
d = dq.agg(F.sum(F.col("dq_alert").cast("int")).alias("alerts"),
           F.sum("rows_quarantined").alias("quarantined")).first()
run = {
    "run_at": dt.datetime.utcnow().isoformat(timespec="seconds"),
    "latest_data_date": str(latest_data),
    "latest_score_date": str(risk.agg(F.max("score_date")).first()[0]),
    "model_version": str(risk.agg(F.max("model_version")).first()[0]),
    "silver_rows": silver.count(),
    "rows_quarantined": int(d.quarantined),
    "dq_alert_days": int(d.alerts),
    "score_days": int(len(drift)),
    "max_psi": float(drift["psi_vs_reference"].max()),
    "drift_alert_days": int(drift["drift_alert"].sum()),
    "labeled_days": int(p.days),
    "queued_drive_days": int(p.q),
    "model_queue_hits": int(p.m),
    "rule_queue_hits": int(p.r),
    "model_queue_precision": float(p.m / p.q),
    "rule_queue_precision": float(p.r / p.q),
}
spark.createDataFrame([run]).write.mode("append").option("mergeSchema", "true").saveAsTable(MON_RUNS)
print(json.dumps(run, indent=2))
dbutils.notebook.exit(json.dumps(run))
