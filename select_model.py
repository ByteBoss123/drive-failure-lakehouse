"""Model selection on the May-June validation window only, then one final test evaluation.

Objective = the business decision: how many failing drives a 10-per-day replacement queue
(7-day cooldown) catches. The test window (Jul 1 - Sep 24) is not used for any choice.
"""
import itertools
import json
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from lakehouse.model import NEG_SAMPLE_PCT, SEED, TRAIN_END, VAL_START, embargo_end, train_and_evaluate
from lakehouse.monitoring import build_queue, drive_level_catch
from lakehouse.transforms import FEATURE_COLS_V1, FEATURE_COLS_V2, LABEL_HORIZON_DAYS

ROOT = Path(__file__).parent
cols = list(dict.fromkeys(["serial_number", "date", "label_fail_7d", "sample_bucket", "counter_sum"] + FEATURE_COLS_V2))
gold = DeltaTable(str(ROOT / "lake" / "gold_drive_features")).to_pandas(columns=cols)
for c in FEATURE_COLS_V2:
    gold[c] = gold[c].astype("float32")
gold["date"] = pd.to_datetime(gold["date"])
gold = gold.sort_values(["serial_number", "date"], kind="stable").reset_index(drop=True)
data_end = str(gold["date"].max().date())


def queue_eval(frame, score_col):
    f = frame.rename(columns={"date": "score_date"})
    f["q"] = build_queue(f, score_col)
    return drive_level_catch(f, "q")


# Validation split: train on rows whose labels resolve before VAL_START; validate May 1 - Jun 24.
dev = gold[gold["date"] <= embargo_end(VAL_START)]
dev = dev[(dev.label_fail_7d == 1) | (dev.sample_bucket < NEG_SAMPLE_PCT)]
val = gold[(gold["date"] >= VAL_START) & (gold["date"] <= embargo_end(TRAIN_END))].copy()
grid = {"features": {"v1": FEATURE_COLS_V1, "v2": FEATURE_COLS_V2},
        "max_leaf_nodes": [15, 31, 63], "learning_rate": [0.05, 0.1]}
results = []
for (fname, fcols), leaves, lr in itertools.product(grid["features"].items(), grid["max_leaf_nodes"],
                                                    grid["learning_rate"]):
    params = dict(max_iter=300, learning_rate=lr, max_leaf_nodes=leaves, l2_regularization=1.0, random_state=SEED)
    m = HistGradientBoostingClassifier(**params).fit(dev[fcols], dev["label_fail_7d"])
    v = val.assign(score=m.predict_proba(val[fcols])[:, 1])
    res = {"features": fname, "max_leaf_nodes": leaves, "learning_rate": lr,
           "val_pr_auc": float(average_precision_score(v.label_fail_7d, v.score)), **queue_eval(v, "score")}
    results.append(res)
    print(json.dumps(res))
val_rule = queue_eval(val, "counter_sum")
best = max(results, key=lambda r: (r["failed_drives_caught"], r["val_pr_auc"]))
print("validation rule:", val_rule)
print("selected:", best)

# One final evaluation on the untouched test window with the selected configuration,
# and with the original configuration under the same protocol for comparison.
final = {}
for name, fcols, params in [
    ("original_v1", FEATURE_COLS_V1, dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=SEED)),
    ("selected", grid["features"][best["features"]],
     dict(max_iter=300, learning_rate=best["learning_rate"], max_leaf_nodes=best["max_leaf_nodes"],
          l2_regularization=1.0, random_state=SEED)),
]:
    _, met = train_and_evaluate(gold, data_end, feature_cols=fcols, params=params)
    t = met.pop("test_scores")
    final[name] = {"model_roc_auc": met["model_roc_auc"], "model_pr_auc": met["model_pr_auc"],
                   "precision_top500": met["model_precision_top500"], **queue_eval(t, "score")}
test_end = gold["date"].max() - pd.Timedelta(days=LABEL_HORIZON_DAYS - 1)
test = gold[(gold["date"] >= TRAIN_END) & (gold["date"] <= test_end)]
final["rule_counter_sum"] = queue_eval(test, "counter_sum")
out = {"validation_window": [VAL_START, str(embargo_end(TRAIN_END).date())], "grid": results,
       "validation_rule": val_rule, "selected": best, "test": final}
(ROOT / "reports" / "selection.json").write_text(json.dumps(out, indent=2, default=str))
print(json.dumps(final, indent=2))
