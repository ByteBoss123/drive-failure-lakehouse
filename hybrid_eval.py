"""Choose the model/rule slot split for the daily queue on the May-June validation window,
then evaluate the chosen split once on the untouched Jul 1 - Sep 24 test window."""
import json
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable
from sklearn.ensemble import HistGradientBoostingClassifier

from lakehouse.model import NEG_SAMPLE_PCT, PARAMS, TRAIN_END, VAL_START, embargo_end, train_and_evaluate
from lakehouse.monitoring import QUEUE_SIZE, build_hybrid_queue, drive_level_catch
from lakehouse.transforms import FEATURE_COLS, LABEL_HORIZON_DAYS

ROOT = Path(__file__).parent
cols = list(dict.fromkeys(["serial_number", "date", "label_fail_7d", "sample_bucket", "counter_sum"] + FEATURE_COLS))
gold = DeltaTable(str(ROOT / "lake" / "gold_drive_features")).to_pandas(columns=cols)
for c in FEATURE_COLS:
    gold[c] = gold[c].astype("float32")
gold["date"] = pd.to_datetime(gold["date"])
gold = gold.sort_values(["serial_number", "date"], kind="stable").reset_index(drop=True)


def evaluate(frame, k):
    f = frame.rename(columns={"date": "score_date"})
    f["q"] = build_hybrid_queue(f, "score", "counter_sum", model_slots=k)
    return drive_level_catch(f, "q")


# Validation: model trained only on labels resolved before May 1, scored on May 1 - Jun 23.
dev = gold[gold["date"] <= embargo_end(VAL_START)]
dev = dev[(dev.label_fail_7d == 1) | (dev.sample_bucket < NEG_SAMPLE_PCT)]
val = gold[(gold["date"] >= VAL_START) & (gold["date"] <= embargo_end(TRAIN_END))].copy()
m = HistGradientBoostingClassifier(**PARAMS).fit(dev[FEATURE_COLS], dev["label_fail_7d"])
val["score"] = m.predict_proba(val[FEATURE_COLS])[:, 1]
validation = {k: evaluate(val, k) for k in range(QUEUE_SIZE + 1)}
for k, r in validation.items():
    print("val model_slots", k, r["failed_drives_caught"], "of", r["failed_drives"])
best_k = max(validation, key=lambda k: (validation[k]["failed_drives_caught"], k))
print("selected model_slots:", best_k)

# Test: the production model (trained with the embargo before Jul 1), evaluated once.
_, met = train_and_evaluate(gold, str(gold["date"].max().date()))
test = met.pop("test_scores")
test_end = gold["date"].max() - pd.Timedelta(days=LABEL_HORIZON_DAYS - 1)
test = test.merge(gold[["serial_number", "date", "counter_sum"]], on=["serial_number", "date"])
test = test[test["date"] <= test_end]
results = {"selected_model_slots": best_k,
           "validation": {str(k): v for k, v in validation.items()},
           "test": {"model_only (10/0)": evaluate(test, QUEUE_SIZE),
                    f"hybrid ({best_k}/{QUEUE_SIZE - best_k})": evaluate(test, best_k),
                    "rule_only (0/10)": evaluate(test, 0)}}
(ROOT / "reports" / "hybrid_selection.json").write_text(json.dumps(results, indent=2))
print(json.dumps(results["test"], indent=2))
