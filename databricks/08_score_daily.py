# Databricks notebook source
# MAGIC %md
# MAGIC # 08 Score: daily drive-failure risk and replacement queue
# MAGIC Loads the latest version of the Unity Catalog model `drive_failure_7d`, scores every
# MAGIC drive for the selected day(s), ranks them, and writes `gold_drive_risk_daily`. The top
# MAGIC `QUEUE_SIZE` drives per day form the replacement queue for the operations team; a queued
# MAGIC drive is not re-queued for `COOLDOWN_DAYS` days. `rule_in_queue` is the same-size queue
# MAGIC ranked by the SMART error counters, kept for comparison.
# MAGIC
# MAGIC `mode = latest` scores the newest data day (the daily job). `mode = backfill` scores
# MAGIC every day from `TRAIN_END` on, so monitoring has history. Reruns replace the same days.

# COMMAND ----------

# MAGIC %pip install scikit-learn==1.9.1 mlflow
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
import mlflow
from mlflow import MlflowClient
from pyspark.sql import functions as F

dbutils.widgets.dropdown("mode", "latest", ["latest", "backfill"])
mode = dbutils.widgets.get("mode")

mlflow.set_registry_uri("databricks-uc")
versions = MlflowClient().search_model_versions(f"name='{MODEL_NAME}'")
if not versions:
    raise RuntimeError(f"No registered versions of {MODEL_NAME}; run 06_train_mlflow first")
version = max(int(v.version) for v in versions)
# The trusted skops types were recorded when the model was logged, so loading needs no extra flags.
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")
print(f"loaded {MODEL_NAME} version {version}")

# COMMAND ----------

gold = spark.table(GOLD_FEATURES)
latest = gold.agg(F.max("date")).first()[0]
first = latest if mode == "latest" else F.lit(TRAIN_END).cast("date")
cols = list(dict.fromkeys(["serial_number", "date", "any_error_counter_nonzero"] + COUNTERS + FEATURE_COLS))
features = gold.filter((F.col("date") >= first) & (F.col("date") <= F.lit(latest))).select(*cols).toPandas()
# Cooldown memory: drives queued in the previous COOLDOWN_DAYS days are not re-queued.
recent_model, recent_rule = {}, {}
if mode == "latest" and spark.catalog.tableExists(RISK):
    hist = (spark.table(RISK)
                 .filter((F.col("score_date") < F.lit(latest))
                         & (F.col("score_date") >= F.date_sub(F.lit(latest), COOLDOWN_DAYS)))
                 .select("score_date", "serial_number", "in_queue", "rule_in_queue").toPandas())
    for col, store in (("in_queue", recent_model), ("rule_in_queue", recent_rule)):
        for row in hist[hist[col]].sort_values("score_date").itertuples():
            store[row.serial_number] = row.score_date
risk = score_frame(features, model, FEATURE_COLS, model_version=version,
                   recent_model=recent_model, recent_rule=recent_rule)
risk_df = (spark.createDataFrame(risk)
                .withColumn("score_date", F.col("score_date").cast("date"))
                .withColumn("scored_at", F.current_timestamp()))
start, end = str(risk["score_date"].min()), str(risk["score_date"].max())

if mode == "backfill" or not spark.catalog.tableExists(RISK):
    risk_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(RISK)
else:
    (risk_df.write.mode("overwrite")
            .option("replaceWhere", f"score_date >= '{start}' AND score_date <= '{end}'")
            .saveAsTable(RISK))
spark.sql(f"COMMENT ON TABLE {RISK} IS 'Daily 7-day failure risk per drive; in_queue marks the "
          f"top {QUEUE_SIZE} drives per day for proactive replacement.'")

# COMMAND ----------

queue = spark.table(RISK).filter((F.col("score_date") == F.lit(end)) & F.col("in_queue")).orderBy("risk_rank")
display(queue.select("score_date", "risk_rank", "serial_number", "risk_score", "counter_sum", "rule_flag"))
out = {"mode": mode, "model_version": version, "days_scored": int(risk["score_date"].nunique()),
       "rows_scored": int(len(risk)), "first_day": start, "last_day": end,
       "table_rows": spark.table(RISK).count()}
print(json.dumps(out))
dbutils.notebook.exit(json.dumps(out))
