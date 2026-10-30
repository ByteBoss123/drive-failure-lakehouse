# Drive Failure Lakehouse (Delta Lake, PySpark, Databricks)

A medallion lakehouse over real Backblaze hard-drive SMART telemetry: incremental Bronze to
Silver to Gold Delta tables, data-quality quarantine, idempotent MERGE, OPTIMIZE/Z-ORDER,
time travel, Unity Catalog governance, and a 7-day drive-failure model tracked in MLflow.

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

All transforms live in `lakehouse/transforms.py` as PySpark functions shared by the local run
and the Databricks notebooks. They are ANSI-safe (regex-gated date parsing, `try_cast`) because
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
* Reproducibility fix: the first Databricks run disagreed with the local model on PR-AUC
  (0.133 vs 0.128) despite identical training rows. The cause was row order: scikit-learn's
  automatic early stopping holds out a validation split by row position, and Spark and delta-rs
  return rows in different orders. Input is now sorted by `(serial_number, date)` before training.

## Run locally

```bash
pip install -r requirements.txt
# put the 273 daily CSVs in data/raw_daily/ (see data/raw_manifest.csv)
python -m pytest -q tests          # 4 tests on the transform rules
python run_local.py                # bronze -> silver -> gold, writes reports/pipeline_metrics.json
python train_local.py              # model + MLflow (sqlite:///mlflow.db), writes reports/model_metrics.json
```

## Run on Databricks (Free Edition)

Add this repository as a Git folder, then run the notebooks in `databricks/` in order:

1. `01_setup_and_download`: schema, Volume, download and checksum the 273 files.
2. `02_bronze_autoloader`: Auto Loader with `availableNow`; run `initial`, then `next_day` or
   `all_remaining`.
3. `03_silver_merge`: streaming `foreachBatch` MERGE, quarantine, replay check, OPTIMIZE,
   `DESCRIBE HISTORY`, `VERSION AS OF`.
4. `04_gold`: feature and fleet tables.
5. `05_governance`: table/column comments, tags, `GRANT SELECT` on gold only, lineage in
   Catalog Explorer.
6. `06_train_mlflow`: same model code, logged to MLflow and registered in Unity Catalog.
7. `07_reconcile`: compares Databricks counts and metrics with the local reference JSON.

**Status:** run end to end on Databricks Free Edition (serverless, Unity Catalog) on 2026-10-30.

## Limitations

* One drive model, nine months, and a third-party extract of the Backblaze data.
* About 148 failed drives in the test window, so the drive-level comparisons are small samples.
* No hyperparameter tuning; one fixed configuration.
