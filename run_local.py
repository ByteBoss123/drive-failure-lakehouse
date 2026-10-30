"""Local reference run of the drive-failure lakehouse.

PySpark runs the same transforms the Databricks notebooks use; Delta tables are
written with delta-rs (the open Delta Lake protocol) because this sandbox
cannot download the Delta Spark jars. Results land in reports/reference_metrics.json
so the Databricks run can be reconciled against them.

Load pattern (simulates daily file drops):
  * initial batch: 2018-01-01 .. 2018-08-31 files
  * incremental:   each September file merged one day at a time
  * replay:        one September day re-merged to prove idempotency
"""
import json
import shutil
import time
from pathlib import Path

import pyarrow.dataset as ds
from deltalake import DeltaTable, write_deltalake
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from lakehouse.transforms import (
    RAW_COLS,
    add_bronze_metadata,
    bronze_to_silver,
    gold_fleet_daily,
    silver_to_gold_features,
)

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw_daily"
LAKE = ROOT / "lake"
TMP = ROOT / "lake_tmp"
REPORTS = ROOT / "reports"
REPLAY_DAY = "2018-09-15"


def spark_session():
    return (SparkSession.builder.master("local[2]").appName("drive-lakehouse")
            .config("spark.driver.memory", "3g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.ansi.enabled", "true")          # match Databricks serverless
            .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


def to_arrow(df, name):
    """Materialize a Spark DataFrame to Parquet and open it as a lazy Arrow dataset."""
    out = TMP / name
    df.write.mode("overwrite").parquet(str(out))
    return ds.dataset(str(out), format="parquet")


def stream(dataset):
    """Stream batches into delta-rs instead of loading everything into memory."""
    return dataset.scanner(batch_size=200_000).to_reader()


def read_delta(spark, path):
    return spark.read.parquet(*DeltaTable(str(path)).file_uris())


def main():
    for p in (LAKE, TMP):
        shutil.rmtree(p, ignore_errors=True)
    REPORTS.mkdir(exist_ok=True)
    spark = spark_session()
    schema = T.StructType([T.StructField(c, T.StringType()) for c in RAW_COLS])
    files = sorted(RAW.glob("*.csv"))
    initial = [str(f) for f in files if f.stem < "2018-09-01"]
    september = [str(f) for f in files if f.stem >= "2018-09-01"]
    m = {"source_files": len(files), "initial_files": len(initial), "incremental_files": len(september)}

    bronze_p, silver_p, quar_p = LAKE / "bronze_smart", LAKE / "silver_smart", LAKE / "silver_quarantine"

    def load_batch(paths, tag):
        bronze = add_bronze_metadata(spark.read.csv(paths, header=True, schema=schema))
        b_arrow = to_arrow(bronze, f"bronze_{tag}")
        write_deltalake(str(bronze_p), stream(b_arrow), mode="append")
        bdf = spark.read.parquet(str(TMP / f"bronze_{tag}"))
        silver, quarantine = bronze_to_silver(bdf)
        s_arrow = to_arrow(silver, f"silver_{tag}")
        q_arrow = to_arrow(quarantine, f"quar_{tag}")
        n_q = q_arrow.count_rows()
        if n_q:
            write_deltalake(str(quar_p), stream(q_arrow), mode="append")
        return b_arrow.count_rows(), s_arrow, n_q

    t0 = time.time()
    n_b, s_arrow, n_q = load_batch(initial, "initial")
    write_deltalake(str(silver_p), stream(s_arrow), mode="overwrite")
    m["initial"] = {"bronze_rows": n_b, "silver_rows": s_arrow.count_rows(), "quarantined_rows": n_q,
                    "seconds": round(time.time() - t0, 1)}
    m["silver_version_after_initial"] = DeltaTable(str(silver_p)).version()

    merges = []
    for path in september:
        day = Path(path).stem
        n_b, s_arrow, n_q = load_batch([path], day.replace("-", ""))
        res = (DeltaTable(str(silver_p))
               .merge(source=s_arrow.to_table(),
                      predicate=f"t.serial_number = s.serial_number AND t.date = s.date AND t.date = '{day}'",
                      source_alias="s", target_alias="t")
               .when_matched_update_all(predicate="t._row_hash <> s._row_hash")
               .when_not_matched_insert_all()
               .execute())
        merges.append({"day": day, "bronze_rows": n_b, "quarantined": n_q,
                       "inserted": res["num_target_rows_inserted"], "updated": res["num_target_rows_updated"]})
    m["incremental_merges"] = merges
    m["incremental_inserted_total"] = sum(x["inserted"] for x in merges)

    # Replay one day: identical file again must change nothing.
    _, s_arrow, _ = load_batch([str(RAW / f"{REPLAY_DAY}.csv")], "replay")
    res = (DeltaTable(str(silver_p))
           .merge(source=s_arrow.to_table(),
                  predicate=f"t.serial_number = s.serial_number AND t.date = s.date AND t.date = '{REPLAY_DAY}'",
                  source_alias="s", target_alias="t")
           .when_matched_update_all(predicate="t._row_hash <> s._row_hash")
           .when_not_matched_insert_all()
           .execute())
    m["replay"] = {"day": REPLAY_DAY, "source_rows": s_arrow.count_rows(),
                   "inserted": res["num_target_rows_inserted"], "updated": res["num_target_rows_updated"]}

    # OPTIMIZE with Z-ORDER, time travel, vacuum dry run.
    dt = DeltaTable(str(silver_p))
    files_before = len(dt.file_uris())
    opt = dt.optimize.z_order(["serial_number"])
    dt = DeltaTable(str(silver_p))
    m["optimize"] = {"files_before": files_before, "files_after": len(dt.file_uris()),
                     "files_added": opt.get("numFilesAdded"), "files_removed": opt.get("numFilesRemoved")}
    v_init = m["silver_version_after_initial"]
    m["time_travel"] = {
        "version_after_initial": v_init,
        "rows_at_that_version": DeltaTable(str(silver_p), version=v_init).to_pyarrow_dataset().count_rows(),
        "rows_now": dt.to_pyarrow_dataset().count_rows(),
        "current_version": dt.version(),
    }
    m["vacuum_dry_run_files"] = len(dt.vacuum(retention_hours=0, enforce_retention_duration=False, dry_run=True))

    # Gold tables from the current silver snapshot.
    silver_df = read_delta(spark, silver_p)
    m["silver_rows"] = silver_df.count()
    m["silver_drives"] = silver_df.select("serial_number").distinct().count()
    m["silver_failures"] = silver_df.filter("failure = 1").count()
    feats = silver_to_gold_features(silver_df)
    write_deltalake(str(LAKE / "gold_drive_features"), stream(to_arrow(feats, "gold_features")), mode="overwrite")
    fleet = gold_fleet_daily(silver_df)
    write_deltalake(str(LAKE / "gold_fleet_daily"), stream(to_arrow(fleet, "gold_fleet")), mode="overwrite")
    gf = DeltaTable(str(LAKE / "gold_drive_features")).to_pyarrow_dataset()
    m["gold_feature_rows"] = gf.count_rows()
    m["gold_positive_label_rows"] = gf.count_rows(filter=ds.field("label_fail_7d") == 1)

    q = DeltaTable(str(quar_p)).to_pyarrow_table(columns=["dq_reason"]).to_pandas()
    m["quarantine_rows_appended"] = int(len(q))
    m["quarantine_by_reason"] = q["dq_reason"].value_counts().to_dict()
    qk = DeltaTable(str(quar_p)).to_pyarrow_table(columns=["serial_number", "date_raw"]).to_pandas()
    m["quarantine_unique_rows"] = int(len(qk.drop_duplicates()))
    b = DeltaTable(str(bronze_p))
    m["bronze_rows_total"] = b.to_pyarrow_dataset().count_rows()
    m["bronze_versions"] = b.version() + 1

    (REPORTS / "pipeline_metrics.json").write_text(json.dumps(m, indent=2, default=str))
    print(json.dumps({k: v for k, v in m.items() if k != "incremental_merges"}, indent=2, default=str))
    spark.stop()
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
