# Databricks notebook source
# MAGIC %md
# MAGIC # 06 Train: 7-day failure model, tracked in MLflow and registered in Unity Catalog
# MAGIC Same `train_and_evaluate` as the local run. Spark pre-filters to the exact rows the model
# MAGIC uses (all positives plus the hash-sampled 5% of negatives for training, every labeled
# MAGIC test row), so the pandas step fits in Free Edition memory.

# COMMAND ----------

# MAGIC %pip install scikit-learn==1.9.1 mlflow
# MAGIC %restart_python

# COMMAND ----------

# MAGIC %run ./00_config

# COMMAND ----------

import json
import mlflow
from mlflow.models import infer_signature
from pyspark.sql import functions as F

gold = spark.table(GOLD_FEATURES)
data_end = str(gold.agg(F.max("date")).first()[0])
last_labeled = F.date_sub(F.lit(data_end).cast("date"), LABEL_HORIZON_DAYS - 1)
rows = gold.filter(
    (F.col("date") <= last_labeled) & (
        (F.col("date") >= TRAIN_END)
        | (F.col("label_fail_7d") == 1)
        | (F.col("sample_bucket") < NEG_SAMPLE_PCT)))
cols = list(dict.fromkeys(["serial_number", "date", "label_fail_7d", "sample_bucket"] + COUNTERS + FEATURE_COLS))
pdf = rows.select(*cols).toPandas()
for c in FEATURE_COLS:
    pdf[c] = pdf[c].astype("float32")
print(len(pdf), "rows pulled for training and evaluation")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)
registered = None
with mlflow.start_run(run_name="hgb_temporal_split") as run:
    model, metrics = train_and_evaluate(pdf, data_end)
    mlflow.log_params({**PARAMS, "train_end": TRAIN_END, "neg_sample_pct": NEG_SAMPLE_PCT})
    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    sample = pdf[FEATURE_COLS].head(100)
    trusted = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]
    try:
        info = mlflow.sklearn.log_model(model, name="model",
                                        signature=infer_signature(sample, model.predict_proba(sample)),
                                        registered_model_name=MODEL_NAME, skops_trusted_types=trusted)
        registered = info.registered_model_version
    except Exception as e:
        registered = f"register failed: {str(e).splitlines()[0][:200]}"
        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=trusted)
    metrics["mlflow_run_id"] = run.info.run_id

spark.createDataFrame([(k, float(v)) for k, v in metrics.items() if isinstance(v, (int, float))],
                      "metric string, value double").write.mode("overwrite").saveAsTable(f"{NS}.model_metrics")
metrics["registered_model_version"] = registered
print(json.dumps(metrics, indent=2, default=str))
dbutils.notebook.exit(json.dumps(metrics, default=str))
