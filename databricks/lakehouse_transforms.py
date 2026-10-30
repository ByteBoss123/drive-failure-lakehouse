# Databricks notebook source
# Generated from lakehouse/transforms.py by scripts/build_shared_notebooks.py. Do not edit.
"""Bronze -> Silver -> Gold transforms for Backblaze SMART telemetry.

Pure PySpark DataFrame functions so the same logic runs in local Spark
(tested) and in Databricks notebooks (Delta tables in Unity Catalog).
Everything is ANSI-safe (regex-gated to_date, try_cast) because Databricks
serverless runs with spark.sql.ansi.enabled=true.
"""
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

KEY_COLS = ["serial_number", "date"]
SMART_COLS = [
    "smart_5_raw",    # reallocated sectors count
    "smart_9_raw",    # power-on hours
    "smart_187_raw",  # reported uncorrectable errors
    "smart_188_raw",  # command timeout
    "smart_193_raw",  # load cycle count
    "smart_194_raw",  # temperature (C)
    "smart_197_raw",  # current pending sector count
    "smart_198_raw",  # offline uncorrectable sectors
    "smart_241_raw",  # total LBAs written
    "smart_242_raw",  # total LBAs read
]
RAW_COLS = ["serial_number", "date", "model", "failure"] + SMART_COLS
# SMART counters Backblaze reports as the strongest failure signals
ERROR_COUNTERS = ["smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw"]
# Features the production model uses (v1, selected on the May-June validation window).
FEATURE_COLS = (
    ["smart_9_raw", "smart_194_raw", "smart_188_raw", "smart_193_raw"]
    + ERROR_COUNTERS
    + [f"{c}_delta_7d" for c in ERROR_COUNTERS]
    + ["temp_mean_7d", "any_error_counter_nonzero"]
)
FEATURE_COLS_V1 = FEATURE_COLS
# v2 candidate: adds 30-day error trends and how persistently errors appeared. Evaluated in
# select_model.py and not selected (it did not catch more failing drives on validation).
FEATURE_COLS_V2 = (
    FEATURE_COLS_V1
    + [f"{c}_delta_30d" for c in ERROR_COUNTERS]
    + ["counter_sum", "counter_sum_delta_7d", "smart_188_raw_delta_7d",
       "error_days_7d", "error_days_30d", "days_since_first_error", "temp_max_7d"]
)
LABEL_HORIZON_DAYS = 7


def add_bronze_metadata(df: DataFrame) -> DataFrame:
    """Keep every raw column as a string and record lineage columns."""
    return (
        df.select(*[F.col(c).cast("string").alias(c) for c in RAW_COLS],
                  F.col("_metadata.file_path").alias("_source_file"))
          .withColumn("_ingested_at", F.current_timestamp())
    )


def _parse_date(col):
    iso = F.col(col).rlike(r"^\d{4}-\d{2}-\d{2}$")
    slash = F.col(col).rlike(r"^\d{4}/\d{1,2}/\d{1,2}$")
    return (F.when(iso, F.to_date(F.col(col), "yyyy-MM-dd"))
             .when(slash, F.to_date(F.col(col), "yyyy/M/d")))


def bronze_to_silver(bronze: DataFrame):
    """Type, validate and deduplicate bronze rows.

    Returns (silver, quarantine). Quarantine rows carry a dq_reason and are
    excluded from silver. Duplicates on (serial_number, date) keep the most
    recently ingested row.
    """
    typed = bronze.select(
        F.trim("serial_number").alias("serial_number"),
        _parse_date("date").alias("date"),
        F.col("date").alias("date_raw"),
        F.trim("model").alias("model"),
        F.expr("try_cast(failure as int)").alias("failure"),
        *[F.expr(f"try_cast({c} as double)").alias(c) for c in SMART_COLS],
        "_source_file", "_ingested_at",
    )
    reason = (
        F.when(F.col("serial_number").isNull() | (F.col("serial_number") == ""), "missing_serial")
         .when(F.col("date").isNull(), "unparseable_date")
         .when(~F.col("failure").isin(0, 1) | F.col("failure").isNull(), "invalid_failure_flag")
         .when(F.col("smart_9_raw").isNull(), "missing_power_on_hours")
         .when(F.greatest(*[F.coalesce(F.col(c), F.lit(0.0)) * -1 for c in SMART_COLS]) > 0, "negative_counter")
    )
    # An implausible temperature is a sensor fault, not a bad row: quarantining it
    # would also drop that drive's failure event. Null the reading and flag it.
    temp_bad = (F.col("smart_194_raw") < 0) | (F.col("smart_194_raw") > 100)
    typed = (typed.withColumn("temp_sensor_invalid", F.coalesce(temp_bad, F.lit(False)))
                  .withColumn("smart_194_raw", F.when(temp_bad, F.lit(None).cast("double"))
                                                .otherwise(F.col("smart_194_raw"))))
    checked = typed.withColumn("dq_reason", reason)
    quarantine = checked.filter(F.col("dq_reason").isNotNull())
    valid = checked.filter(F.col("dq_reason").isNull()).drop("dq_reason")

    w = Window.partitionBy(*KEY_COLS).orderBy(F.col("_ingested_at").desc(), F.col("_source_file").desc())
    silver = (
        valid.withColumn("_rn", F.row_number().over(w))
             .filter("_rn = 1").drop("_rn", "date_raw")
             .withColumn("_row_hash", F.sha2(F.concat_ws("|", *[
                 F.coalesce(F.col(c).cast("string"), F.lit("")) for c in ["model", "failure"] + SMART_COLS]), 256))
    )
    return silver, quarantine


def silver_to_gold_features(silver: DataFrame) -> DataFrame:
    """Per drive-day features plus the 7-day-ahead failure label."""
    day = F.datediff(F.col("date"), F.lit("1970-01-01"))
    w7 = Window.partitionBy("serial_number").orderBy(day).rangeBetween(-(LABEL_HORIZON_DAYS - 1), 0)
    w30 = Window.partitionBy("serial_number").orderBy(day).rangeBetween(-29, 0)
    w_all = Window.partitionBy("serial_number").orderBy(day).rangeBetween(Window.unboundedPreceding, 0)
    fail_date = (silver.filter("failure = 1").groupBy("serial_number")
                       .agg(F.min("date").alias("failure_date")))
    feats = silver.select("serial_number", "date", "model", "failure", *SMART_COLS)
    for c in ERROR_COUNTERS:
        feats = (feats.withColumn(f"{c}_delta_7d", F.col(c) - F.min(c).over(w7))
                      .withColumn(f"{c}_delta_30d", F.col(c) - F.min(c).over(w30)))
    err = F.greatest(*[(F.col(c) > 0).cast("int") for c in ERROR_COUNTERS])
    csum = sum(F.coalesce(F.col(c), F.lit(0.0)) for c in ERROR_COUNTERS)
    feats = (
        feats.withColumn("temp_mean_7d", F.avg("smart_194_raw").over(w7))
             .withColumn("temp_max_7d", F.max("smart_194_raw").over(w7))
             .withColumn("any_error_counter_nonzero", err)
             .withColumn("counter_sum", csum)
             .withColumn("counter_sum_delta_7d", F.col("counter_sum") - F.min("counter_sum").over(w7))
             .withColumn("smart_188_raw_delta_7d", F.col("smart_188_raw") - F.min("smart_188_raw").over(w7))
             .withColumn("error_days_7d", F.sum("any_error_counter_nonzero").over(w7))
             .withColumn("error_days_30d", F.sum("any_error_counter_nonzero").over(w30))
             .withColumn("days_since_first_error",
                         F.coalesce(day - F.min(F.when(F.col("any_error_counter_nonzero") == 1, day)).over(w_all),
                                    F.lit(-1)))
             .join(fail_date, "serial_number", "left")
             .withColumn("days_to_failure", F.datediff("failure_date", "date"))
             .withColumn("label_fail_7d",
                         (F.col("days_to_failure").between(0, LABEL_HORIZON_DAYS - 1)).cast("int"))
             .fillna({"label_fail_7d": 0})
             # deterministic 0-99 bucket so Spark (Databricks) and pandas (local)
             # draw the identical negative-class training sample
             .withColumn("sample_bucket",
                         F.pmod(F.xxhash64("serial_number", F.col("date").cast("string")), F.lit(100)).cast("int"))
    )
    return feats


def gold_fleet_daily(silver: DataFrame) -> DataFrame:
    """Daily fleet health: active drives, failures, daily failure rate."""
    return (silver.groupBy("date")
                  .agg(F.countDistinct("serial_number").alias("active_drives"),
                       F.sum("failure").alias("failures"),
                       F.avg("smart_194_raw").alias("avg_temp_c"))
                  .withColumn("daily_failure_rate", F.col("failures") / F.col("active_drives"))
                  .orderBy("date"))
