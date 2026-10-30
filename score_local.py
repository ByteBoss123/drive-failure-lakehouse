"""Local reference for batch scoring and monitoring (mirrors notebooks 08 and 09).

Scores every drive-day from TRAIN_END onward with the deterministic model, builds the
daily replacement queue, then computes queue precision and score drift.
Writes reports/scoring_metrics.json for the Databricks reconciliation.
"""
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from deltalake import DeltaTable

from lakehouse.model import TRAIN_END, train_and_evaluate
from lakehouse.monitoring import QUEUE_SIZE, queue_performance_daily, score_drift_daily, score_frame
from lakehouse.transforms import FEATURE_COLS, LABEL_HORIZON_DAYS
from run_local import spark_session

ROOT = Path(__file__).parent
cols = ["serial_number", "date", "label_fail_7d", "sample_bucket", "smart_5_raw", "smart_187_raw",
        "smart_197_raw", "smart_198_raw"] + FEATURE_COLS
gold = DeltaTable(str(ROOT / "lake" / "gold_drive_features")).to_pandas(columns=list(dict.fromkeys(cols)))
for c in FEATURE_COLS:
    gold[c] = gold[c].astype("float32")
data_end = str(gold["date"].max())
model, _ = train_and_evaluate(gold, data_end)

window = gold[gold["date"].astype(str) >= TRAIN_END]
risk = score_frame(window, model, FEATURE_COLS, model_version="local")
tmp = ROOT / "lake_tmp" / "risk"
shutil.rmtree(tmp, ignore_errors=True)
tmp.mkdir(parents=True)
pq.write_table(pa.Table.from_pandas(risk, preserve_index=False), tmp / "risk.parquet")

spark = spark_session()
risk_df = spark.read.parquet(str(tmp))
gold_df = spark.createDataFrame(
    gold.loc[gold["date"].astype(str) >= TRAIN_END, ["date", "serial_number", "label_fail_7d"]])
last_labeled = (gold["date"].max() - __import__("datetime").timedelta(days=LABEL_HORIZON_DAYS - 1))
perf = queue_performance_daily(risk_df, gold_df, last_labeled).toPandas()
drift = score_drift_daily(risk_df)

m = {
    "score_days": int(risk["score_date"].nunique()),
    "risk_rows": int(len(risk)),
    "queue_size_per_day": QUEUE_SIZE,
    "labeled_days": int(len(perf)),
    "last_labeled_date": str(last_labeled),
    "queued_drive_days": int(perf["queue_size"].sum()),
    "model_queue_hits": int(perf["model_hits"].sum()),
    "rule_queue_hits": int(perf["rule_hits"].sum()),
    "model_queue_precision": float(perf["model_hits"].sum() / perf["queue_size"].sum()),
    "rule_queue_precision": float(perf["rule_hits"].sum() / perf["queue_size"].sum()),
    "max_psi": float(drift["psi_vs_reference"].max()),
    "drift_alert_days": int(drift["drift_alert"].sum()),
}
(ROOT / "reports" / "scoring_metrics.json").write_text(json.dumps(m, indent=2))
print(json.dumps(m, indent=2))
spark.stop()
shutil.rmtree(tmp, ignore_errors=True)
