"""Batch scoring helpers and pipeline / model monitoring.

Shared by the Databricks scoring (08) and monitoring (09) notebooks and covered by
tests/test_monitoring.py. Spark functions return small per-day aggregates.
"""
import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

QUEUE_SIZE = 10                 # drives the ops team can inspect per day
COOLDOWN_DAYS = 7               # a queued drive is not re-queued for this many days (it was just inspected)
PSI_BINS = 10
PSI_ALERT = 0.2                 # common rule of thumb: > 0.2 is a material shift
QUARANTINE_RATE_ALERT = 0.001   # alert if > 0.1% of a day's rows are rejected


COUNTERS = ["smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw"]


def score_frame(features: pd.DataFrame, model, feature_cols, model_version: str,
                recent_model: dict = None, recent_rule: dict = None) -> pd.DataFrame:
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
    return rank_daily(scored, recent_model=recent_model, recent_rule=recent_rule)


def build_queue(df: pd.DataFrame, score_col: str, queue_size: int = QUEUE_SIZE,
                cooldown_days: int = COOLDOWN_DAYS, recent: dict = None) -> pd.Series:
    """Daily queue: the top `queue_size` drives by `score_col`, skipping any drive queued in the
    previous `cooldown_days` days. `recent` maps serial_number -> last queued date (for daily runs).

    Ties are broken by serial_number so the queue is deterministic.
    """
    last = dict(recent or {})
    in_queue = pd.Series(False, index=df.index)
    order = df.sort_values(["score_date", score_col, "serial_number"],
                           ascending=[True, False, True], kind="stable")
    for day, rows in order.groupby("score_date", sort=True):
        day = pd.Timestamp(day)
        picked = 0
        for idx, serial in zip(rows.index, rows["serial_number"], strict=True):
            prev = last.get(serial)
            if prev is not None and (day - pd.Timestamp(prev)).days < cooldown_days:
                continue
            in_queue.at[idx] = True
            last[serial] = day
            picked += 1
            if picked == queue_size:
                break
    return in_queue


def _take(ranked: pd.DataFrame, n: int, day: pd.Timestamp, picked: list, last: dict, cooldown_days: int):
    """Append up to `n` total picks from `ranked`, skipping picked or cooling-down drives."""
    for idx, serial in zip(ranked.index, ranked["serial_number"], strict=True):
        if len(picked) >= n:
            return
        prev = last.get(serial)
        if idx in picked or (prev is not None and (day - pd.Timestamp(prev)).days < cooldown_days):
            continue
        picked.append(idx)
        last[serial] = day


def build_hybrid_queue(df: pd.DataFrame, model_col: str, rule_col: str, model_slots: int,
                       queue_size: int = QUEUE_SIZE, cooldown_days: int = COOLDOWN_DAYS,
                       recent: dict = None) -> pd.Series:
    """Daily queue mixing two rankings: `model_slots` drives from the model ranking, the rest from
    the rule ranking, with one shared cooldown. If one ranking runs out, the other fills the slots."""
    last = dict(recent or {})
    in_queue = pd.Series(False, index=df.index)
    for day, rows in df.groupby("score_date", sort=True):
        day = pd.Timestamp(day)
        by_model = rows.sort_values([model_col, "serial_number"], ascending=[False, True], kind="stable")
        by_rule = rows.sort_values([rule_col, "serial_number"], ascending=[False, True], kind="stable")
        picked = []
        _take(by_model, model_slots, day, picked, last, cooldown_days)
        _take(by_rule, queue_size, day, picked, last, cooldown_days)
        _take(by_model, queue_size, day, picked, last, cooldown_days)
        in_queue.loc[picked] = True
    return in_queue


def rank_daily(scored: pd.DataFrame, queue_size: int = QUEUE_SIZE, recent_model: dict = None,
               recent_rule: dict = None) -> pd.DataFrame:
    """Rank drives within each score_date and mark the model queue and an equal-size rule queue."""
    out = scored.sort_values(["score_date", "risk_score", "serial_number"],
                             ascending=[True, False, True], kind="stable").copy()
    out["risk_rank"] = out.groupby("score_date").cumcount() + 1
    out["in_queue"] = build_queue(out, "risk_score", queue_size, recent=recent_model)
    if "counter_sum" in out:
        out["rule_in_queue"] = build_queue(out, "counter_sum", queue_size, recent=recent_rule)
    return out


def drive_level_catch(df: pd.DataFrame, queue_col: str) -> dict:
    """Distinct drives queued, and how many drives that failed were queued in their final 7 days."""
    q = df[df[queue_col]]
    failed = df.loc[df["label_fail_7d"] == 1, "serial_number"].nunique()
    caught = q.loc[q["label_fail_7d"] == 1, "serial_number"].nunique()
    return {"distinct_drives_queued": int(q["serial_number"].nunique()), "failed_drives": int(failed),
            "failed_drives_caught": int(caught), "catch_rate": caught / failed if failed else 0.0,
            "queued_drive_hit_rate": float(q.groupby("serial_number")["label_fail_7d"].max().mean())}


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
    ranked by the summed counters under the same cooldown policy.
    """
    labels = gold.select(F.col("date").alias("score_date"), "serial_number", "label_fail_7d")
    labeled = (risk.filter(F.col("score_date") <= F.lit(last_labeled_date))
                   .join(labels, ["score_date", "serial_number"], "left")
                   .fillna({"label_fail_7d": 0}))
    rule = (labeled.filter("rule_in_queue")
                   .groupBy("score_date").agg(F.sum("label_fail_7d").alias("rule_hits")))
    model = (labeled.filter("in_queue").groupBy("score_date")
                    .agg(F.count("*").alias("queue_size"), F.sum("label_fail_7d").alias("model_hits")))
    return (model.join(rule, "score_date", "left").fillna({"rule_hits": 0})
                 .withColumn("model_precision", F.col("model_hits") / F.col("queue_size"))
                 .withColumn("rule_precision", F.col("rule_hits") / F.col("queue_size"))
                 .orderBy("score_date"))
