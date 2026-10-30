# Databricks notebook source
# MAGIC %md
# MAGIC # 07 Reconcile Databricks against the local reference run
# MAGIC Compares row counts and model metrics with `reports/*.json` produced by `run_local.py`
# MAGIC and `train_local.py`. Pipeline counts must match exactly. Model metrics should match
# MAGIC closely (same rows and seed; small drift is possible across library builds).
# MAGIC Run after loading all 273 files (02 with `all_remaining`, then 03, 04, 06).

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
import pandas as pd

ref = json.load(open(os.path.join(REPO_ROOT, "reports", "pipeline_metrics.json")))
ref_model = json.load(open(os.path.join(REPO_ROOT, "reports", "model_metrics.json")))

silver = spark.table(SILVER)
gold = spark.table(GOLD_FEATURES)
checks = [
    ("silver_rows", ref["silver_rows"], silver.count()),
    ("silver_drives", ref["silver_drives"], silver.select("serial_number").distinct().count()),
    ("silver_failures", ref["silver_failures"], silver.filter("failure = 1").count()),
    ("quarantined_rows", ref["quarantine_unique_rows"],
     spark.table(QUARANTINE).select("serial_number", "date_raw").distinct().count()),
    ("gold_feature_rows", ref["gold_feature_rows"], gold.count()),
    ("gold_positive_label_rows", ref["gold_positive_label_rows"], gold.filter("label_fail_7d = 1").count()),
]
db_model = {r.metric: r.value for r in spark.table(f"{NS}.model_metrics").collect()}
for k in ("test_positive_rows", "model_roc_auc", "model_pr_auc", "model_recall_at_budget", "rule_recall"):
    checks.append((k, ref_model[k], db_model.get(k)))

result = pd.DataFrame(checks, columns=["check", "local_reference", "databricks"])
result["match"] = [
    (abs(a - b) <= 0.005 if isinstance(a, float) else a == b) if b is not None else False
    for a, b in zip(result.local_reference, result.databricks)]
display(result)
assert result.match.all(), "Mismatch: see table above"
