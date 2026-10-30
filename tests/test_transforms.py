import datetime as dt

import pytest
from pyspark.sql import SparkSession

from lakehouse.transforms import SMART_COLS, bronze_to_silver, gold_fleet_daily, silver_to_gold_features


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[1]")
         .config("spark.sql.ansi.enabled", "true")
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    yield s
    s.stop()


def bronze_rows(spark, rows):
    cols = ["serial_number", "date", "model", "failure"] + SMART_COLS + ["_source_file", "_ingested_at"]
    out = []
    for r in rows:
        base = {c: "0" for c in SMART_COLS}
        base.update({"smart_9_raw": "100", "smart_194_raw": "25", "model": "ST4000DM000",
                     "failure": "0", "_source_file": "f.csv",
                     "_ingested_at": dt.datetime(2018, 10, 1)})
        base.update(r)
        out.append(tuple(base[c] for c in cols))
    return spark.createDataFrame(out, cols)


def test_both_date_formats_parse(spark):
    b = bronze_rows(spark, [{"serial_number": "A", "date": "2018/1/1"},
                            {"serial_number": "A", "date": "2018-01-02"}])
    silver, quar = bronze_to_silver(b)
    assert quar.count() == 0
    assert sorted(r.date for r in silver.collect()) == [dt.date(2018, 1, 1), dt.date(2018, 1, 2)]


def test_quarantine_reasons(spark):
    b = bronze_rows(spark, [
        {"serial_number": "", "date": "2018-01-01"},
        {"serial_number": "B", "date": "01-01-2018"},
        {"serial_number": "C", "date": "2018-01-01", "smart_9_raw": None},
        {"serial_number": "D", "date": "2018-01-01", "smart_5_raw": "-3"},
        {"serial_number": "E", "date": "2018-01-01", "smart_194_raw": "250", "failure": "1"},
        {"serial_number": "F", "date": "2018-01-01", "failure": "x"},
    ])
    silver, quar = bronze_to_silver(b)
    reasons = {r.serial_number: r.dq_reason for r in quar.collect()}
    assert reasons == {"": "missing_serial", "B": "unparseable_date", "C": "missing_power_on_hours",
                       "D": "negative_counter", "F": "invalid_failure_flag"}
    # Bad temperature: row and its failure event are kept, reading nulled and flagged.
    e = silver.collect()
    assert len(e) == 1 and e[0].serial_number == "E" and e[0].failure == 1
    assert e[0].smart_194_raw is None and e[0].temp_sensor_invalid is True


def test_dedupe_keeps_latest_ingest(spark):
    b = bronze_rows(spark, [
        {"serial_number": "A", "date": "2018-01-01", "smart_5_raw": "1",
         "_ingested_at": dt.datetime(2018, 10, 1)},
        {"serial_number": "A", "date": "2018-01-01", "smart_5_raw": "7",
         "_ingested_at": dt.datetime(2018, 10, 2)},
    ])
    silver, _ = bronze_to_silver(b)
    rows = silver.collect()
    assert len(rows) == 1 and rows[0].smart_5_raw == 7.0


def test_label_and_delta(spark):
    days = [f"2018-01-{d:02d}" for d in range(1, 11)]
    rows = [{"serial_number": "A", "date": d, "smart_5_raw": str(i)} for i, d in enumerate(days)]
    rows[-1]["failure"] = "1"                                   # fails on 2018-01-10
    rows += [{"serial_number": "B", "date": d} for d in days]   # healthy
    silver, _ = bronze_to_silver(bronze_rows(spark, rows))
    g = silver_to_gold_features(silver)
    a = {r.date.day: r for r in g.filter("serial_number = 'A'").collect()}
    assert [d for d in a if a[d].label_fail_7d == 1] == [4, 5, 6, 7, 8, 9, 10]
    assert a[10].smart_5_raw_delta_7d == 6.0                    # 9 - 3 over a 7-day window
    assert a[1].smart_5_raw_delta_7d == 0.0
    assert g.filter("serial_number = 'B' AND label_fail_7d = 1").count() == 0
    fleet = {r.date.day: r for r in gold_fleet_daily(silver).collect()}
    assert fleet[10].failures == 1 and fleet[10].active_drives == 2
