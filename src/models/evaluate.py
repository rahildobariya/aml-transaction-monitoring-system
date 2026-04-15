import json
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.pipeline.tiering import (
    assign_tiers,
    evaluate_tiers,
    print_tier_report,
)


def _load_params() -> dict:
    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


def _flatten_metrics(metrics: dict) -> dict[str, float]:
    # mlflow.log_metrics only takes a flat dict, so we need to unpack the nested structure
    flat: dict[str, float] = {}

    for tier, tier_metrics in metrics["per_tier"].items():
        prefix = f"test_{tier.lower()}"
        flat[f"{prefix}_n_alerts"]  = tier_metrics["n_alerts"]
        flat[f"{prefix}_n_fraud"]   = tier_metrics["n_fraud"]
        flat[f"{prefix}_precision"] = tier_metrics["precision"]

    for k, v in metrics["combined"].items():
        flat[f"test_{k}"] = v

    flat["test_auc_roc"] = metrics["ranking"]["auc_roc"]
    flat["test_auc_pr"]  = metrics["ranking"]["auc_pr"]

    flat["test_total_transactions"] = metrics["volume"]["total_transactions"]
    flat["test_total_fraud"]        = metrics["volume"]["total_fraud"]
    flat["test_fraud_rate"]         = metrics["volume"]["fraud_rate"]

    return flat


def evaluate() -> dict:
    params     = _load_params()
    mlflow_cfg = params["mlflow"]
    tier_cfg   = params["alerts"]["tier_k"]

    feat_cfg     = params["features"]
    feature_cols = feat_cfg["numeric"] + feat_cfg.get("categorical", [])

    test_path = ROOT / "data" / "processed" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            "Run `python src/features/engineering.py` first."
        )

    print(f"[evaluate] Loading {test_path} ...")
    test_df = pd.read_csv(test_path)

    # only use features that actually exist in the test file
    feature_cols = [c for c in feature_cols if c in test_df.columns]
    print(f"[evaluate] {len(feature_cols)} features, {len(test_df):,} transactions")

    model_path = ROOT / "models" / "xgboost_aml.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run `python src/models/train.py` first."
        )

    artifact = joblib.load(model_path)
    model    = artifact["model"]
    run_id   = artifact.get("run_id")

    print(f"[evaluate] Model loaded  (scale_pos_weight={artifact.get('scale_pos_weight', 'N/A')})")
    print(f"[evaluate] Tier config   : "
          f"CRITICAL={tier_cfg['critical_k']}  "
          f"HIGH={tier_cfg['high_k']}  "
          f"MEDIUM={tier_cfg['medium_k']}")

    print("[evaluate] Scoring all transactions and assigning tiers ...")
    df_tiered = assign_tiers(
        test_df,
        model=model,
        feature_cols=feature_cols,
        tier_cfg=tier_cfg,
        label_col="is_fraud",
    )

    metrics = evaluate_tiers(df_tiered, label_col="is_fraud")
    print_tier_report(metrics)

    reports_dir  = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "test_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved -> {metrics_path}")

    # also save the scored transactions so the dashboard can load them
    scored_path  = reports_dir / "test_scored.csv"
    cols_to_save = [c for c in ["fraud_score", "rank", "tier", "is_fraud"] if c in df_tiered.columns]
    df_tiered[cols_to_save].to_csv(scored_path, index=False)
    print(f"[evaluate] Scored output -> {scored_path}")

    if run_id:
        flat_metrics = _flatten_metrics(metrics)
        mlflow.set_tracking_uri((ROOT / mlflow_cfg["tracking_uri"]).as_uri())
        mlflow.set_experiment(mlflow_cfg["experiment_name"])
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(flat_metrics)
            mlflow.log_artifact(str(metrics_path))
        print(f"[evaluate] Metrics logged to MLflow run: {run_id}")

    return metrics


if __name__ == "__main__":
    evaluate()
