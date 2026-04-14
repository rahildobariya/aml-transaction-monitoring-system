"""
AML Model Evaluation -- Tiered Scoring Architecture
======================================================
Replaces the previous binary Top-K evaluation with a rank-based,
tier-driven evaluation that mirrors production AML operations.

Scoring flow:
  1. Load test set  (never seen during training or threshold selection)
  2. Score every transaction with model.predict_proba -> fraud_score
  3. Rank all transactions globally by fraud_score (1 = highest risk)
  4. Assign tiers by rank using business-owned cutoffs from params.yaml:
       CRITICAL : top critical_k ranks          -> full investigator review
       HIGH     : next high_k ranks             -> automated hold + priority Q
       MEDIUM   : next medium_k ranks           -> soft flag / step-up auth
       LOW      : all remaining                 -> no action
  5. Report:
       - Precision per tier            (investigator efficiency)
       - Recall for CRITICAL + HIGH    (primary safety metric)
       - Alert volume per tier         (operational load)
       - AUC-ROC / AUC-PR             (threshold-independent ranking quality)

No fixed probability thresholds (0.5, 0.9, etc.) appear anywhere here.
No Top-K binary cut. One model, one global score, one ranking.

Usage:
    python src/models/evaluate.py
"""

from __future__ import annotations

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
    """
    Flatten nested metrics dict into a single-level dict for MLflow logging.

    MLflow log_metrics requires a flat {str: float} mapping.
    Nested keys are joined with underscores.

    Parameters
    ----------
    metrics : dict
        Output of evaluate_tiers (nested structure).

    Returns
    -------
    dict[str, float]
        Flat metrics dict safe for mlflow.log_metrics.
    """
    flat: dict[str, float] = {}

    # Per-tier metrics
    for tier, tier_metrics in metrics["per_tier"].items():
        prefix = f"test_{tier.lower()}"
        flat[f"{prefix}_n_alerts"]  = tier_metrics["n_alerts"]
        flat[f"{prefix}_n_fraud"]   = tier_metrics["n_fraud"]
        flat[f"{prefix}_precision"] = tier_metrics["precision"]

    # Combined CRITICAL + HIGH
    for k, v in metrics["combined"].items():
        flat[f"test_{k}"] = v

    # Ranking quality
    flat["test_auc_roc"] = metrics["ranking"]["auc_roc"]
    flat["test_auc_pr"]  = metrics["ranking"]["auc_pr"]

    # Volume
    flat["test_total_transactions"] = metrics["volume"]["total_transactions"]
    flat["test_total_fraud"]        = metrics["volume"]["total_fraud"]
    flat["test_fraud_rate"]         = metrics["volume"]["fraud_rate"]

    return flat


def evaluate() -> dict:
    """
    Run tiered evaluation on the held-out test set.

    Returns
    -------
    dict
        Full nested metrics dict (same structure as evaluate_tiers output).
    """
    params    = _load_params()
    mlflow_cfg = params["mlflow"]
    tier_cfg   = params["alerts"]["tier_k"]

    feat_cfg     = params["features"]
    feature_cols = feat_cfg["numeric"] + feat_cfg.get("categorical", [])

    # ------------------------------------------------------------------
    # 1. Load test data
    # ------------------------------------------------------------------
    test_path = ROOT / "data" / "processed" / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            "Run `python src/features/engineering.py` first."
        )

    print(f"[evaluate] Loading {test_path} ...")
    test_df = pd.read_csv(test_path)

    # Restrict feature list to columns that exist in the test set
    feature_cols = [c for c in feature_cols if c in test_df.columns]
    print(f"[evaluate] {len(feature_cols)} features, {len(test_df):,} transactions")

    # ------------------------------------------------------------------
    # 2. Load model artifact
    # ------------------------------------------------------------------
    model_path = ROOT / "models" / "xgboost_aml.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run `python src/models/train.py` first."
        )

    artifact = joblib.load(model_path)
    model  = artifact["model"]
    run_id = artifact.get("run_id")

    print(f"[evaluate] Model loaded  (scale_pos_weight={artifact.get('scale_pos_weight', 'N/A')})")
    print(f"[evaluate] Tier config   : "
          f"CRITICAL={tier_cfg['critical_k']}  "
          f"HIGH={tier_cfg['high_k']}  "
          f"MEDIUM={tier_cfg['medium_k']}")

    # ------------------------------------------------------------------
    # 3. Score + rank + assign tiers
    # ------------------------------------------------------------------
    print("[evaluate] Scoring all transactions and assigning tiers ...")
    df_tiered = assign_tiers(
        test_df,
        model=model,
        feature_cols=feature_cols,
        tier_cfg=tier_cfg,
        label_col="is_fraud",
    )

    # ------------------------------------------------------------------
    # 4. Compute metrics
    # ------------------------------------------------------------------
    metrics = evaluate_tiers(df_tiered, label_col="is_fraud")

    # ------------------------------------------------------------------
    # 5. Print structured report
    # ------------------------------------------------------------------
    print_tier_report(metrics)

    # ------------------------------------------------------------------
    # 6. Save metrics to disk (DVC metrics file)
    # ------------------------------------------------------------------
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "test_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics saved -> {metrics_path}")

    # Save tiered scored output (useful for dashboard and monitoring)
    scored_path = reports_dir / "test_scored.csv"
    cols_to_save = ["fraud_score", "rank", "tier", "is_fraud"]
    cols_to_save = [c for c in cols_to_save if c in df_tiered.columns]
    df_tiered[cols_to_save].to_csv(scored_path, index=False)
    print(f"[evaluate] Scored output -> {scored_path}")

    # ------------------------------------------------------------------
    # 7. Log to MLflow (parent training run)
    # ------------------------------------------------------------------
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
