# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Reconcile Databricks against the local reference run
# MAGIC Pipeline counts must match exactly; model and scoring metrics within 0.005. Scoring checks
# MAGIC run only after 08 has backfilled the test window and 09 has written the queue table.

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
import pandas as pd
from pyspark.sql import functions as F

silver = spark.table(SILVER)
gold = spark.table(GOLD_FEATURES)
checks = [
    ("silver_rows", REF["silver_rows"], silver.count()),
    ("silver_drives", REF["silver_drives"], silver.select("serial_number").distinct().count()),
    ("silver_failures", REF["silver_failures"], silver.filter("failure = 1").count()),
    ("quarantined_rows", REF["quarantine_unique_rows"],
     spark.table(QUARANTINE).select("serial_number", "date_raw").distinct().count()),
    ("gold_feature_rows", REF["gold_feature_rows"], gold.count()),
    ("gold_positive_label_rows", REF["gold_positive_label_rows"], gold.filter("label_fail_7d = 1").count()),
]
db_model = {r.metric: r.value for r in spark.table(f"{NS}.model_metrics").collect()}
for k in ("test_positive_rows", "model_roc_auc", "model_pr_auc", "model_recall_at_budget",
          "rule_recall", "model_precision_top500", "counter_sum_precision_top500"):
    checks.append((k, REF_MODEL[k], db_model.get(k)))

if spark.catalog.tableExists(MON_QUEUE):
    risk = spark.table(RISK).filter(F.col("score_date") >= TRAIN_END)
    q = spark.table(MON_QUEUE).agg(F.sum("queue_size").alias("q"), F.sum("model_hits").alias("m"),
                                   F.sum("rule_hits").alias("r"), F.count("*").alias("d")).first()
    checks += [
        ("score_days", REF_SCORING["score_days"], risk.select("score_date").distinct().count()),
        ("risk_rows", REF_SCORING["risk_rows"], risk.count()),
        ("labeled_days", REF_SCORING["labeled_days"], q.d),
        ("model_queue_hits", REF_SCORING["model_queue_hits"], q.m),
        ("rule_queue_hits", REF_SCORING["rule_queue_hits"], q.r),
    ]

result = pd.DataFrame(checks, columns=["check", "local_reference", "databricks"])
result["match"] = [
    (abs(a - b) <= 0.005 if isinstance(a, float) else a == b) if b is not None else False
    for a, b in zip(result.local_reference, result.databricks)]
display(result)
out = {"all_match": bool(result.match.all()),
       "rows": [[c, l, d, bool(m)] for c, l, d, m in result.itertuples(index=False)]}
dbutils.notebook.exit(json.dumps(out, default=str))
