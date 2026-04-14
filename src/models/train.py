"""
XGBoost AML Model Training Script
====================================
Trains an XGBoost classifier on the processed transaction dataset.

MLOps integration:
  - MLflow: logs all hyperparameters, metrics, and the trained model artifact
  - scale_pos_weight: computed DYNAMICALLY from training data (n_neg / n_pos)
    -- never hardcoded; automatically adapts to any class imbalance ratio
  - Capacity-constrained threshold: alert volume capped to investigator budget
    (target 500-800 alerts per ~40K transactions, configurable in params.yaml)

Usage:
    python src/models/train.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_params() -> dict:
    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Threshold selection
# ---------------------------------------------------------------------------

def capacity_constrained_threshold(
    y_val: np.ndarray,
    y_prob: np.ndarray,
    alert_budget_low: int,
    alert_budget_high: int,
    test_size: int,
) -> tuple[float, dict]:
    """
    Find the threshold that constrains daily alert volume to the investigator budget.

    Strategy: derive the target alert RATE from the budget and test set size,
    apply that rate to the validation set to find the operating threshold.
    This decouples threshold selection from the arbitrary val/test split ratio.

    Parameters
    ----------
    y_val : np.ndarray
        True labels on the validation set.
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    alert_budget_low : int
        Minimum acceptable alerts (floor for investigator queue).
    alert_budget_high : int
        Maximum acceptable alerts (investigator capacity ceiling).
    test_size : int
        Approximate number of transactions scored per period (used to derive
        target alert rate that is then applied to the validation set).

    Returns
    -------
    tuple
        (threshold, metrics_dict)
    """
    # Derive target alert rate from business budget and expected scoring volume
    target_alert_rate = ((alert_budget_low + alert_budget_high) / 2) / test_size

    # Apply that rate to the validation set to find the probability cutoff
    val_budget    = max(1, int(len(y_prob) * target_alert_rate))
    sorted_probs  = np.sort(y_prob)[::-1]
    threshold     = float(sorted_probs[min(val_budget, len(sorted_probs) - 1)])

    y_pred = (y_prob >= threshold).astype(int)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)

    n_alerts = int(y_pred.sum())
    metrics = {
        "threshold": round(threshold, 4),
        "val_precision": round(report["1"]["precision"], 4),
        "val_recall": round(report["1"]["recall"], 4),
        "val_f1": round(report["1"]["f1-score"], 4),
        "val_auc_roc": round(float(roc_auc_score(y_val, y_prob)), 4),
        "val_auc_pr": round(float(average_precision_score(y_val, y_prob)), 4),
        "val_n_alerts": n_alerts,
        "val_alert_rate": round(n_alerts / len(y_prob), 4),
    }

    return threshold, metrics


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train() -> None:
    params        = _load_params()
    data_cfg      = params["data"]
    model_cfg     = params["model"]
    threshold_cfg = params["threshold"]
    mlflow_cfg    = params["mlflow"]

    # 1. Load processed training data
    train_path = ROOT / "data" / "processed" / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Processed data not found at {train_path}. "
            "Run `python src/features/engineering.py` first."
        )

    print(f"[train] Loading {train_path} ...")
    train_df = pd.read_csv(train_path)

    # Enforce feature list from params.yaml -- only use configured columns
    feat_cfg     = params["features"]
    feature_cols = feat_cfg["numeric"] + feat_cfg.get("categorical", [])
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    print(f"[train] Using {len(feature_cols)} features: {feature_cols}")

    X = train_df[feature_cols]
    y = train_df["is_fraud"]

    # 2. Hold out a validation split (from the training data -- no test leakage)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=data_cfg["val_size"],
        stratify=y,
        random_state=data_cfg["random_seed"],
    )

    print(f"[train] Train: {len(X_train):,}  Val: {len(X_val):,}")
    print(f"[train] Fraud rate in train: {y_train.mean():.2%}")

    # 3. Compute scale_pos_weight DYNAMICALLY from training data
    #    This is the correct approach -- never hardcode the imbalance ratio.
    #    XGBoost uses this to up-weight minority class (fraud) in the loss function.
    n_pos = int(y_train.sum())
    n_neg = int((y_train == 0).sum())
    dynamic_spw = round(n_neg / max(n_pos, 1), 2)
    print(f"[train] Class counts -- Fraud: {n_pos:,}  Legit: {n_neg:,}")
    print(f"[train] scale_pos_weight (dynamic): {dynamic_spw}  "
          f"(was {model_cfg['params'].get('scale_pos_weight', 'N/A')} in params.yaml)")

    # 4. Configure MLflow
    mlflow.set_tracking_uri((ROOT / mlflow_cfg["tracking_uri"]).as_uri())
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # Approximate test set size for capacity-constrained threshold calibration
    n_total          = len(train_df) / (1 - data_cfg["test_size"])
    approx_test_size = int(n_total * data_cfg["test_size"])

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[train] MLflow run ID: {run_id}")

        # Build XGBoost params -- override scale_pos_weight with dynamic value
        xgb_params = model_cfg["params"].copy()
        xgb_params.pop("use_label_encoder", None)   # removed in XGBoost >= 1.6
        xgb_params["scale_pos_weight"] = dynamic_spw

        # Log hyperparameters
        mlflow.log_params(xgb_params)
        mlflow.log_params(
            {
                "val_size": data_cfg["val_size"],
                "threshold_strategy": threshold_cfg["strategy"],
                "alert_budget_low": threshold_cfg["alert_budget_low"],
                "alert_budget_high": threshold_cfg["alert_budget_high"],
                "n_features": len(feature_cols),
            }
        )

        # 5. Train XGBoost
        print("[train] Training XGBoost ...")
        model = XGBClassifier(**xgb_params, early_stopping_rounds=50)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # 6. Capacity-constrained threshold on validation set
        print("[train] Finding capacity-constrained threshold ...")
        y_val_prob = model.predict_proba(X_val)[:, 1]
        best_threshold, val_metrics = capacity_constrained_threshold(
            y_val.values,
            y_val_prob,
            alert_budget_low=threshold_cfg["alert_budget_low"],
            alert_budget_high=threshold_cfg["alert_budget_high"],
            test_size=approx_test_size,
        )

        print(f"[train] Threshold        : {best_threshold:.4f}")
        print(f"[train] Val alerts       : {val_metrics['val_n_alerts']:,} "
              f"({val_metrics['val_alert_rate']:.2%} alert rate)")
        print(f"[train] Val precision    : {val_metrics['val_precision']:.4f}")
        print(f"[train] Val recall       : {val_metrics['val_recall']:.4f}")
        print(f"[train] Val F1           : {val_metrics['val_f1']:.4f}")
        print(f"[train] Val AUC-ROC      : {val_metrics['val_auc_roc']:.4f}")
        print(f"[train] Val AUC-PR       : {val_metrics['val_auc_pr']:.4f}")

        mlflow.log_metrics(val_metrics)
        mlflow.log_param("chosen_threshold", best_threshold)
        mlflow.log_param("scale_pos_weight_actual", dynamic_spw)

        # 7. Persist model artifact
        models_dir = ROOT / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "xgboost_aml.pkl"

        artifact = {
            "model": model,
            "threshold": best_threshold,
            "feature_names": list(X.columns),
            "scale_pos_weight": dynamic_spw,
            "run_id": run_id,
        }
        joblib.dump(artifact, model_path)
        print(f"[train] Model saved -> {model_path}")

        mlflow.log_artifact(str(model_path), artifact_path="model")

        # Save val metrics to reports/
        reports_dir = ROOT / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        with open(reports_dir / "val_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)

        mlflow.log_artifact(str(reports_dir / "val_metrics.json"))

    print(f"[train] Done. MLflow experiment: {mlflow_cfg['experiment_name']}")
    print(f"         Run `mlflow ui` in project root to view results.")


if __name__ == "__main__":
    train()
