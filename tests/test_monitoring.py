import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pyspark.sql import SparkSession

from lakehouse.monitoring import (
    QUEUE_SIZE,
    build_hybrid_queue,
    build_queue,
    drive_level_catch,
    psi,
    queue_performance_daily,
    rank_daily,
    score_drift_daily,
)


@pytest.fixture(scope="module")
def spark():
    s = (SparkSession.builder.master("local[1]")
         .config("spark.sql.ansi.enabled", "true")
         .config("spark.sql.session.timeZone", "UTC").getOrCreate())
    yield s
    s.stop()


def test_rank_daily_queue_and_ties():
    d = dt.date(2018, 7, 1)
    df = pd.DataFrame({"score_date": [d] * 12,
                       "serial_number": [f"S{i:02d}" for i in range(12)],
                       "risk_score": [0.95, 0.95] + [0.01 * i for i in range(10)]})
    df.loc[11, "risk_score"] = 0.99
    out = rank_daily(df, queue_size=3)
    q = out[out.in_queue].serial_number.tolist()
    assert q == ["S11", "S00", "S01"]          # 0.99 first, then the 0.95 tie broken by serial
    assert out.in_queue.sum() == 3 and out.risk_rank.max() == 12


def test_cooldown_skips_recently_queued_drives():
    days = [dt.date(2018, 7, 1) + dt.timedelta(days=k) for k in range(9)]
    rows = [{"score_date": d, "serial_number": s, "score": sc}
            for d in days for s, sc in (("A", 0.9), ("B", 0.8), ("C", 0.1))]
    df = pd.DataFrame(rows)
    df["q"] = build_queue(df, "score", queue_size=1, cooldown_days=7)
    picked = df[df.q].set_index("score_date").serial_number.to_dict()
    # A, then B and C while A cools down; nothing is eligible on days 3-6; A returns after 7 days.
    assert [picked.get(d) for d in days] == ["A", "B", "C", None, None, None, None, "A", "B"]


def test_drive_level_catch_counts_distinct_drives():
    df = pd.DataFrame({"serial_number": ["A", "A", "B", "C"], "label_fail_7d": [1, 1, 0, 1],
                       "q": [True, True, True, False]})
    r = drive_level_catch(df, "q")
    assert r["distinct_drives_queued"] == 2 and r["failed_drives"] == 2
    assert r["failed_drives_caught"] == 1 and r["queued_drive_hit_rate"] == 0.5


def test_psi_identical_is_zero_and_shift_is_positive():
    a = [100] * 10
    assert psi(a, a) == pytest.approx(0.0)
    assert psi(a, [10] * 5 + [190] * 5) > 0.2


def _risk(spark, rows):
    return spark.createDataFrame(pd.DataFrame(rows))


def test_queue_performance_counts_hits_against_labels(spark):
    d1, d2 = dt.date(2018, 7, 1), dt.date(2018, 7, 2)
    risk = []
    for d in (d1, d2):
        for i in range(QUEUE_SIZE + 5):
            risk.append({"score_date": d, "serial_number": f"S{i}", "risk_score": 1.0 - i / 100,
                         "counter_sum": float(i), "in_queue": i < QUEUE_SIZE,
                         "rule_in_queue": i >= 5})
    gold = [{"date": d1, "serial_number": "S0", "label_fail_7d": 1},
            {"date": d1, "serial_number": "S14", "label_fail_7d": 1},
            {"date": d2, "serial_number": "S1", "label_fail_7d": 1}]
    perf = {r.score_date: r for r in
            queue_performance_daily(_risk(spark, risk), _risk(spark, gold), d1).collect()}
    assert list(perf) == [d1]                      # d2 is past the last labeled date
    assert perf[d1].queue_size == QUEUE_SIZE
    assert perf[d1].model_hits == 1                # S0 is in the model queue
    assert perf[d1].rule_hits == 1                 # S14 is in the rule queue


def test_score_drift_flags_shifted_day(spark):
    rng = np.random.default_rng(0)
    rows = []
    for k in range(8):
        d = dt.date(2018, 7, 1) + dt.timedelta(days=k)
        scores = rng.uniform(0, 0.5, 2000) if k < 7 else rng.uniform(0.5, 1.0, 2000)
        rows += [{"score_date": d, "serial_number": f"S{i}", "risk_score": float(s)}
                 for i, s in enumerate(scores)]
    out = score_drift_daily(_risk(spark, rows), reference_days=7)
    assert out.drift_alert.tolist() == [False] * 7 + [True]


def test_hybrid_queue_splits_slots_and_shares_cooldown():
    d = dt.date(2018, 7, 1)
    df = pd.DataFrame({"score_date": [d] * 4, "serial_number": ["A", "B", "C", "D"],
                       "m": [0.9, 0.8, 0.1, 0.0], "r": [0.0, 0.0, 5.0, 9.0]})
    q = build_hybrid_queue(df, "m", "r", model_slots=1, queue_size=2)
    assert df[q].serial_number.tolist() == ["A", "D"]     # top model drive + top rule drive
    recent = {"A": pd.Timestamp(d) - pd.Timedelta(days=1)}  # A is cooling down
    q2 = build_hybrid_queue(df, "m", "r", model_slots=1, queue_size=2, recent=recent)
    assert sorted(df[q2].serial_number.tolist()) == ["B", "D"]
