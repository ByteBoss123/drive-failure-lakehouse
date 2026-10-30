"""Train and evaluate the failure model on the local gold table, tracked in MLflow."""
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from deltalake import DeltaTable

from lakehouse.model import NEG_SAMPLE_PCT, PARAMS, TRAIN_END, train_and_evaluate
from lakehouse.transforms import FEATURE_COLS

TRUSTED = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]

ROOT = Path(__file__).parent
cols = ["serial_number", "date", "label_fail_7d", "sample_bucket"] + FEATURE_COLS
df = DeltaTable(str(ROOT / "lake" / "gold_drive_features")).to_pandas(columns=cols)
for c in FEATURE_COLS:
    df[c] = df[c].astype("float32")
data_end = str(df["date"].max())

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
mlflow.set_experiment("drive-failure-7d")
with mlflow.start_run(run_name="hgb_temporal_split") as run:
    model, metrics = train_and_evaluate(df, data_end)
    metrics.pop("test_scores")
    mlflow.log_params({**PARAMS, "train_end": TRAIN_END, "neg_sample_pct": NEG_SAMPLE_PCT,
                       "features": ",".join(FEATURE_COLS)})
    mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    # Model trained in this run, so its tree predictor type is trusted for skops serialization.
    mlflow.sklearn.log_model(model, name="model", skops_trusted_types=TRUSTED)
    metrics["mlflow_run_id"] = run.info.run_id

(ROOT / "reports" / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
