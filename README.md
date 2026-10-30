# Drive Failure Lakehouse (Delta Lake, PySpark, Databricks)

An end-to-end data and ML system for proactive hard-drive replacement, built on real Backblaze
SMART telemetry: incremental Bronze/Silver/Gold Delta tables with data-quality quarantine, a
7-day failure model tracked in MLflow and registered in Unity Catalog, daily batch scoring into
a replacement queue, monitoring tables, and a scheduled Databricks Workflow. CI runs the tests on
every push.

**Business question:** an operations team can inspect or swap only a few drives a day. Which
ones should be first in line, so drives are replaced before they fail?

## Data

* **Source:** Backblaze Drive Stats daily snapshots for the Seagate ST4000DM000 fleet,
  2018-01-01 to 2018-09-30 (273 daily files, 12 SMART columns), taken from the
  [Phillweston/Disk-Failure-Prediction](https://github.com/Phillweston/Disk-Failure-Prediction)
  GitHub mirror at commit `d7e8e59`. Backblaze's own download host was not reachable from the
  build environment, so the mirror was used; every file's SHA-256 is in `data/raw_manifest.csv`.
* **Scope:** one drive model and a column subset chosen by the mirror's author, not the full
  Backblaze fleet.
* **Size:** 7,835,606 drive-day rows, 32,164 drives, 460 failures.

## Architecture

| Layer | Table | What it holds |
| --- | --- | --- |
| Bronze | `bronze_smart` | Raw rows as strings plus source file and load time. Append-only. |
| Silver | `silver_smart` | Typed, validated, deduplicated drive-days. Key `(serial_number, date)`. |
| Silver | `silver_quarantine` | Rows that failed data-quality rules, with `dq_reason`. |
| Gold | `gold_drive_features` | 7-day SMART deltas, temperature mean, label `fails within 7 days`. |
| Gold | `gold_fleet_daily` | Active drives, failures and failure rate per day. |
| Gold | `gold_drive_risk_daily` | Daily risk score and rank per drive; `in_queue` marks the top 10. |
| Monitor | `monitor_dq_daily` | Rows kept and quarantined per day, sensor faults, DQ alert flag. |
| Monitor | `monitor_score_daily` | Score distribution and PSI drift against the first 7 scored days. |
| Monitor | `monitor_queue_daily` | Queued drives that actually failed, model vs an equal-size rule queue. |
| Monitor | `monitor_run_log` | One row per pipeline run with headline numbers and alert counts. |

**Daily Databricks Workflow** (`jobs/daily_pipeline.json`, 6:30 AM ET):
`bronze_ingest -> silver_merge -> gold_features -> score_drives -> monitor`

All logic lives in the `lakehouse/` package (`transforms.py`, `model.py`, `monitoring.py`),
covered by unit tests. `scripts/build_shared_notebooks.py` generates the Databricks shared-code
notebooks from it, and CI fails if they drift. Transforms are ANSI-safe (regex-gated date parsing, `try_cast`) because
Databricks serverless enables ANSI mode.

## Verified results (local reference run)

Locally, PySpark 3.5.3 runs the transforms and Delta tables are written with delta-rs 1.6.6,
because the Delta Spark jars could not be downloaded in the build environment. Every figure below
comes from `reports/pipeline_metrics.json` and `reports/model_metrics.json`.

**Pipeline**

* Loaded 243 files (Jan to Aug, 7,088,609 rows) as the initial batch, then merged 30 September
  files one day at a time: 746,997 rows inserted, 0 updated.
* Replaying 2018-09-15 (24,874 rows) through the same MERGE inserted 0 and updated 0 rows,
  because matched rows update only when their content hash changes.
* Reconciled exactly: 7,835,509 silver rows + 97 quarantined rows = 7,835,606 raw rows.
* OPTIMIZE with Z-ORDER on `serial_number` compacted silver from 36 files to 4.
* Time travel: version 0 returns the 7,088,512-row initial load; the current version (31)
  returns 7,835,509 rows.

**Data-quality findings**

* The 2018-01-01 file uses `2018/1/1` dates while the other 272 files use `2018-01-02`;
  both formats are parsed.
* 97 rows have no SMART values at all (mostly on 2018-08-14, 2018-07-25 and 2018-05-02);
  they are quarantined as `missing_power_on_hours`.
* Drive W3019Q2H reports a constant 141 C for 67 days and then fails on 2018-06-01. An early
  version quarantined out-of-range temperatures and silently dropped this failure (459 of 460).
  The rule now nulls the reading, sets `temp_sensor_invalid`, and keeps the row, so all 460
  failures reach silver.

**Model: predict failure within 7 days** (`lakehouse/model.py`)

* Temporal split: train on Jan to Jun, test on Jul 1 to Sep 24 (the last 6 days are excluded
  because their 7-day labels are not yet observable).
* Training uses all 2,141 positive rows and a deterministic 5% hash sample of negatives
  (275,421 rows). Testing uses all 2,220,122 test drive-days with their real 0.044% positive rate.
* HistGradientBoosting: ROC-AUC 0.767, PR-AUC 0.126. Ranking drives by the sum of the four
  SMART error counters gives a similar ROC-AUC (0.772) but a PR-AUC of only 0.024.
* With a small alert budget the model is far more precise: in its top 500 test alerts,
  28.8% precede a failure within 7 days, against 4.8% for the counter-sum ranking.
* With a broad budget it does not beat the simple rule. Alerting on any nonzero error counter
  flags 63,420 drive-days and catches 56.3% of failure rows (98 of 148 failed drives). The model
  at the same budget catches 47.0% (94 drives).
* Takeaway: the model is useful for a short, high-confidence replacement queue. The rule is
  better for broad coverage. The two are complementary, not a replacement.
**Daily replacement queue** (`lakehouse/monitoring.py`, `reports/scoring_metrics.json`)

* Every drive is scored each day from Jul 1 to Sep 30 (92 days, 2,368,010 drive-days), and the
  top 10 form that day's queue.
* Over the 86 days whose 7-day outcome is known, 188 of 860 queued drive-days (21.9%) failed
  within 7 days. An equal-size daily queue ranked by the SMART error counters caught 29 (3.4%).
* Score drift stayed below the 0.2 PSI alert threshold on every day (max 0.13).

* Reproducibility fix: the first Databricks run disagreed with the local model on PR-AUC
  (0.133 vs 0.128) despite identical training rows. The cause was row order: scikit-learn's
  automatic early stopping holds out a validation split by row position, and Spark and delta-rs
  return rows in different orders. Input is now sorted by `(serial_number, date)` before training.

## Run locally

```bash
pip install -r requirements.txt
# put the 273 daily CSVs in data/raw_daily/ (see data/raw_manifest.csv)
python -m pytest -q tests          # 8 tests: transforms, ranking, PSI, queue precision, drift
python run_local.py                # bronze -> silver -> gold, writes reports/pipeline_metrics.json
python train_local.py              # model + MLflow (sqlite:///mlflow.db), writes reports/model_metrics.json
python score_local.py              # daily queue + monitoring, writes reports/scoring_metrics.json
python scripts/build_shared_notebooks.py   # regenerate databricks/lakehouse_*.py after code changes
```

## Run on Databricks (Free Edition)

Import the notebooks in `databricks/` into one workspace folder, then run them in order once:

1. `01_setup_and_download`: schema, Volume, download and checksum the 273 files.
2. `02_bronze_autoloader`: Auto Loader with `availableNow`; run `initial`, then `next_day` or
   `all_remaining`.
3. `03_silver_merge`: streaming `foreachBatch` MERGE, quarantine, replay check, OPTIMIZE,
   `DESCRIBE HISTORY`, `VERSION AS OF`.
4. `04_gold`: feature and fleet tables.
5. `05_governance`: table/column comments, tags, `GRANT SELECT` on gold only, lineage in
   Catalog Explorer.
6. `06_train_mlflow`: same model code, logged to MLflow and registered in Unity Catalog.
7. `08_score_daily` with `mode = backfill`: score Jul to Sep into `gold_drive_risk_daily`.
8. `09_monitor`: build the monitoring tables.
9. `07_reconcile`: compares Databricks pipeline, model and scoring numbers with the local reference.

Then create the daily job from `jobs/daily_pipeline.json`. It lands one new file per day
(`next_day`), merges it, rebuilds gold, scores the newest day, and refreshes monitoring.

**Status (2026-10-30):** notebooks 01 to 07 run end to end on Databricks Free Edition (serverless, Unity Catalog), with every pipeline and model check matching the local reference. Scoring (08), monitoring (09) and the scheduled job are being verified.

## Limitations

* One drive model, nine months, and a third-party extract of the Backblaze data.
* About 148 failed drives in the test window, so the drive-level comparisons are small samples.
* No hyperparameter tuning; one fixed configuration.
