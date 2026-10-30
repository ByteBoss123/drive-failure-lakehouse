"""Drive failure model: predict failure within the next 7 days.

Shared by the local reference run and the Databricks training notebook.
Evaluation is on the full, naturally imbalanced test period (no resampling)
and compared with the SMART error-counter rule at the same alert budget.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from lakehouse.transforms import FEATURE_COLS, LABEL_HORIZON_DAYS

TRAIN_END = "2018-07-01"          # train: dates before this
NEG_SAMPLE_PCT = 5                # negatives with sample_bucket < 5 (deterministic hash)
SEED = 42
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
              l2_regularization=1.0, random_state=SEED)


def split(df: pd.DataFrame, data_end: str):
    """Temporal split; drop the last horizon of days whose labels are censored."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    last_labeled = pd.Timestamp(data_end) - pd.Timedelta(days=LABEL_HORIZON_DAYS - 1)
    df = df[df["date"] <= last_labeled]
    train = df[df["date"] < TRAIN_END]
    test = df[df["date"] >= TRAIN_END]
    pos = train[train.label_fail_7d == 1]
    neg = train[(train.label_fail_7d == 0) & (train.sample_bucket < NEG_SAMPLE_PCT)]
    return pd.concat([pos, neg]), test, last_labeled


def _at_budget(y, score, k):
    top = np.argsort(-score, kind="stable")[:k]
    hits = int(y[top].sum())
    return hits, hits / k, hits / int(y.sum())


def train_and_evaluate(df: pd.DataFrame, data_end: str):
    train, test, last_labeled = split(df, data_end)
    model = HistGradientBoostingClassifier(**PARAMS)
    model.fit(train[FEATURE_COLS], train["label_fail_7d"])

    y = test["label_fail_7d"].to_numpy()
    score = model.predict_proba(test[FEATURE_COLS])[:, 1]
    rule = test["any_error_counter_nonzero"].to_numpy()
    k = int(rule.sum())
    rule_hits = int(y[rule == 1].sum())
    m_hits, m_prec, m_rec = _at_budget(y, score, k)

    # Drive level: of drives that failed in the test window, how many got
    # at least one alert in their final 7 days (rule vs model at same budget).
    test = test.assign(score=score, rule=rule)
    thr = np.sort(score)[::-1][k - 1]
    pos_rows = test[test.label_fail_7d == 1]
    failed = pos_rows.serial_number.nunique()
    rule_caught = pos_rows[pos_rows.rule == 1].serial_number.nunique()
    model_caught = pos_rows[pos_rows.score >= thr].serial_number.nunique()

    # Ranked baseline: sum of the four SMART error counters (a stronger rule that can rank).
    counter_sum = test[["smart_5_raw", "smart_187_raw", "smart_197_raw", "smart_198_raw"]].fillna(0).sum(axis=1).to_numpy()
    small_budgets = {}
    for b in (500, 2000):
        mh, mp, mr = _at_budget(y, score, b)
        ch, cp, cr = _at_budget(y, counter_sum, b)
        small_budgets[b] = (mp, mr, cp, cr)

    metrics = {
        "train_rows": int(len(train)), "train_positive_rows": int(train.label_fail_7d.sum()),
        "test_rows": int(len(test)), "test_positive_rows": int(y.sum()),
        "test_base_rate": float(y.mean()), "last_labeled_date": str(last_labeled.date()),
        "model_roc_auc": float(roc_auc_score(y, score)),
        "model_pr_auc": float(average_precision_score(y, score)),
        "rule_pr_auc_equivalent": float(average_precision_score(y, rule)),
        "alert_budget_rows": k,
        "rule_precision": rule_hits / k, "rule_recall": rule_hits / int(y.sum()),
        "model_precision_at_budget": m_prec, "model_recall_at_budget": m_rec,
        "test_failed_drives": int(failed),
        "rule_failed_drives_caught": int(rule_caught),
        "model_failed_drives_caught": int(model_caught),
        "counter_sum_pr_auc": float(average_precision_score(y, counter_sum)),
        "counter_sum_roc_auc": float(roc_auc_score(y, counter_sum)),
    }
    for b, (mp, mr, cp, cr) in small_budgets.items():
        metrics.update({f"model_precision_top{b}": mp, f"model_recall_top{b}": mr,
                        f"counter_sum_precision_top{b}": cp, f"counter_sum_recall_top{b}": cr})
    return model, metrics
