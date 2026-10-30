# Databricks notebook source
# Generated from lakehouse/monitoring.py by scripts/build_shared_notebooks.py. Do not edit.
"""Batch scoring helpers and pipeline / model monitoring.

Shared by the Databricks scoring (08) and monitoring (09) notebooks and covered by
tests/test_monitoring.py. Spark functions return small per-day aggregates.
"""
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

QUEUE_SIZE = 10                 # drives the ops team can inspect per day
PSI_BINS = 10
PSI_ALERT = 0.2                 # common rule of thumb: > 0.2 is a material shift
QUARANTINE_RATE_ALERT = 0.001   # alert if > 0.1% of a day's rows are rejected


COUNTERS = ["smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw"]


def score_frame(features: pd.DataFrame, model, feature_cols, model_version: str) -> pd.DataFrame:
    """Score drive-days and build the daily risk table (one row per drive per day)."""
    f = features.sort_values(["date", "serial_number"], kind="stable").reset_index(drop=True)
    scored = pd.DataFrame({
        "score_date": pd.to_datetime(f["date"]).dt.date,
        "serial_number": f["serial_number"],
        "risk_score": model.predict_proba(f[feature_cols].astype("float32"))[:, 1],
        "counter_sum": f[COUNTERS].fillna(0).sum(axis=1).astype("float64"),
        "rule_flag": f["any_error_counter_nonzero"].fillna(0).astype("int32"),
    })
    scored["model_version"] = str(model_version)
    return rank_daily(scored)


def rank_daily(scored: pd.DataFrame, queue_size: int = QUEUE_SIZE) -> pd.DataFrame:
    """Rank drives within each score_date; the top `queue_size` form the replacement queue.

    Ties are broken by serial_number so the queue is deterministic.
    """
    out = scored.sort_values(["score_date", "risk_score", "serial_number"],
                             ascending=[True, False, True], kind="stable").copy()
    out["risk_rank"] = out.groupby("score_date").cumcount() + 1
    out["in_queue"] = out["risk_rank"] <= queue_size
    return out


def psi(expected_counts, actual_counts, eps: float = 1e-6) -> float:
    """Population stability index between two binned count vectors."""
    e = np.asarray(expected_counts, dtype=float)
    a = np.asarray(actual_counts, dtype=float)
    e = np.clip(e / e.sum(), eps, None)
    a = np.clip(a / a.sum(), eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def dq_daily(silver: DataFrame, quarantine: DataFrame, parse_date) -> DataFrame:
    """Rows kept, rows quarantined, quarantine rate and sensor-fault count per data date."""
    kept = (silver.groupBy("date")
                  .agg(F.count("*").alias("rows_kept"),
                       F.sum(F.col("temp_sensor_invalid").cast("int")).alias("temp_sensor_invalid")))
    rejected = (quarantine.select(parse_date("date_raw").alias("date"), "serial_number", "date_raw")
                          .dropDuplicates(["serial_number", "date_raw"])
                          .groupBy("date").agg(F.count("*").alias("rows_quarantined")))
    return (kept.join(rejected, "date", "full_outer")
                .fillna({"rows_kept": 0, "rows_quarantined": 0, "temp_sensor_invalid": 0})
                .withColumn("quarantine_rate",
                            F.col("rows_quarantined") / (F.col("rows_kept") + F.col("rows_quarantined")))
                .withColumn("dq_alert", F.col("quarantine_rate") > QUARANTINE_RATE_ALERT)
                .orderBy("date"))


def score_drift_daily(risk: DataFrame, reference_days: int = 7) -> pd.DataFrame:
    """Per-day score summary and PSI against the first `reference_days` of scoring."""
    days = [r.score_date for r in risk.select("score_date").distinct().orderBy("score_date").collect()]
    ref = risk.filter(F.col("score_date").isin(days[:reference_days]))
    qs = ref.approxQuantile("risk_score", [i / PSI_BINS for i in range(1, PSI_BINS)], 0.0001)
    # Bin index = number of reference quantiles at or below the score (plain SQL, so it
    # also runs on Databricks serverless, where pyspark.ml transformers may be unavailable).
    edges = sorted(set(qs))
    binned = risk.withColumn("bin", sum((F.col("risk_score") >= F.lit(q)).cast("int") for q in edges))
    counts = (binned.groupBy("score_date", "bin").count().toPandas()
                    .pivot(index="score_date", columns="bin", values="count").fillna(0).sort_index())
    ref_counts = counts.loc[counts.index.isin(days[:reference_days])].sum(axis=0)
    summary = (risk.groupBy("score_date")
                   .agg(F.count("*").alias("drives_scored"),
                        F.avg("risk_score").alias("mean_score"),
                        F.expr("percentile_approx(risk_score, 0.95)").alias("p95_score"))
                   .toPandas().set_index("score_date").sort_index())
    summary["psi_vs_reference"] = [psi(ref_counts, counts.loc[d]) for d in summary.index]
    summary["drift_alert"] = summary["psi_vs_reference"] > PSI_ALERT
    return summary.reset_index()


def queue_performance_daily(risk: DataFrame, gold: DataFrame, last_labeled_date) -> DataFrame:
    """For days whose 7-day outcome is known: how many queued drives actually failed.

    Compares the model queue with an equal-size queue taken from the error-counter rule,
    ranked by the summed counters.
    """
    from pyspark.sql import Window

    labels = gold.select(F.col("date").alias("score_date"), "serial_number", "label_fail_7d")
    labeled = (risk.filter(F.col("score_date") <= F.lit(last_labeled_date))
                   .join(labels, ["score_date", "serial_number"], "left")
                   .fillna({"label_fail_7d": 0}))
    w = Window.partitionBy("score_date").orderBy(F.col("counter_sum").desc(), F.col("serial_number"))
    rule = (labeled.withColumn("rule_rank", F.row_number().over(w))
                   .filter(F.col("rule_rank") <= QUEUE_SIZE)
                   .groupBy("score_date").agg(F.sum("label_fail_7d").alias("rule_hits")))
    model = (labeled.filter("in_queue").groupBy("score_date")
                    .agg(F.count("*").alias("queue_size"), F.sum("label_fail_7d").alias("model_hits")))
    return (model.join(rule, "score_date", "left").fillna({"rule_hits": 0})
                 .withColumn("model_precision", F.col("model_hits") / F.col("queue_size"))
                 .withColumn("rule_precision", F.col("rule_hits") / F.col("queue_size"))
                 .orderBy("score_date"))
