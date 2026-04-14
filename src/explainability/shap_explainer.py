"""
SHAP Explainability & AML Reason Code Generator
=================================================
Generates human-readable 'Reason Codes' for every model alert using SHAP
(SHapley Additive exPlanations) with XGBoost's TreeExplainer.

For each transaction flagged as suspicious, this module:
  1. Computes SHAP values using the fast TreeExplainer
  2. Identifies the top-N features driving the positive prediction
  3. Maps feature names to AML-domain Reason Codes (defined in params.yaml)
  4. Returns a structured explanation suitable for investigator review

Reason Code format:  RC-XX: <human-readable description>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_params() -> dict:
    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# SHAP explainer wrapper
# ---------------------------------------------------------------------------

class AMLExplainer:
    """
    Wraps SHAP TreeExplainer for the AML XGBoost model.

    Attributes
    ----------
    model : XGBClassifier
        The trained XGBoost model.
    threshold : float
        Decision threshold for alert generation.
    feature_names : list[str]
        Ordered list of feature names matching model input.
    reason_code_map : dict[str, str]
        Mapping from feature name → human-readable reason code string.
    top_n : int
        Number of top reason codes to return per transaction.
    explainer : shap.TreeExplainer
        The underlying SHAP explainer.
    """

    def __init__(self, artifact: dict, params: dict) -> None:
        self.model = artifact["model"]
        self.threshold = artifact["threshold"]
        self.feature_names = artifact["feature_names"]

        shap_cfg = params["shap"]
        self.top_n = shap_cfg["top_n_features"]
        self.reason_code_map: dict[str, str] = shap_cfg["reason_codes"]

        self.explainer = shap.TreeExplainer(self.model)

    # ------------------------------------------------------------------
    # Core explanation methods
    # ------------------------------------------------------------------

    def explain_batch(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        """
        Compute SHAP explanations for a batch of transactions.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (columns must match feature_names).

        Returns
        -------
        list[dict]
            One explanation dict per row:
            {
                "risk_score"    : float (0-1),
                "risk_pct"      : str  ("73.4%"),
                "alert_tier"    : str  ("HIGH"),
                "is_alert"      : bool,
                "reason_codes"  : list[str],
                "shap_values"   : dict[feature -> shap_value],
                "top_features"  : list[{"feature", "shap", "direction"}]
            }
        """
        X = X.reindex(columns=self.feature_names, fill_value=0)
        shap_values = self.explainer.shap_values(X)

        probabilities = self.model.predict_proba(X)[:, 1]

        results = []
        for i, prob in enumerate(probabilities):
            row_shap = shap_values[i]
            row_explanation = self._build_explanation(prob, row_shap)
            results.append(row_explanation)

        return results

    def explain_single(self, x: pd.Series) -> dict[str, Any]:
        """
        Explain a single transaction row.

        Parameters
        ----------
        x : pd.Series
            A single transaction's features.

        Returns
        -------
        dict
            Explanation dict (same schema as explain_batch output).
        """
        df = pd.DataFrame([x])
        return self.explain_batch(df)[0]

    def _build_explanation(self, prob: float, shap_row: np.ndarray) -> dict[str, Any]:
        """
        Build the full explanation dict for a single prediction.
        """
        alert_tier = self._get_alert_tier(prob)
        is_alert = prob >= self.threshold

        # Sort features by absolute SHAP value (descending)
        feature_shap = list(zip(self.feature_names, shap_row))
        feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)

        # Build top-N entries
        top_features = []
        reason_codes = []

        for feat_name, shap_val in feature_shap:
            if len(reason_codes) >= self.top_n:
                break
            if shap_val > 0:  # only include features that INCREASE fraud probability
                top_features.append(
                    {
                        "feature": feat_name,
                        "shap": round(float(shap_val), 4),
                        "direction": "increases",
                    }
                )
                rc = self.reason_code_map.get(feat_name)
                if rc:
                    reason_codes.append(rc)

        # Fallback: if no positive SHAP features, take top-N by magnitude
        if not reason_codes:
            for feat_name, shap_val in feature_shap[: self.top_n]:
                rc = self.reason_code_map.get(feat_name, f"RC-UNK: {feat_name} is anomalous")
                reason_codes.append(rc)

        return {
            "risk_score": round(float(prob), 4),
            "risk_pct": f"{prob * 100:.1f}%",
            "alert_tier": alert_tier,
            "is_alert": bool(is_alert),
            "reason_codes": reason_codes,
            "shap_values": {
                f: round(float(v), 4) for f, v in zip(self.feature_names, shap_row)
            },
            "top_features": top_features,
        }

    @staticmethod
    def _get_alert_tier(prob: float) -> str:
        """Map risk probability to investigator alert tier."""
        if prob >= 0.80:
            return "CRITICAL"
        elif prob >= 0.60:
            return "HIGH"
        elif prob >= 0.30:
            return "MEDIUM"
        else:
            return "LOW"

    # ------------------------------------------------------------------
    # Convenience: run on test set and return annotated DataFrame
    # ------------------------------------------------------------------

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotate a DataFrame with risk scores and reason codes.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix. May include 'is_fraud' ground-truth column.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with appended columns:
            risk_score, risk_pct, alert_tier, is_alert, reason_codes
        """
        feature_cols = [c for c in self.feature_names if c in df.columns]
        X = df[feature_cols]

        explanations = self.explain_batch(X)

        df = df.copy()
        df["risk_score"] = [e["risk_score"] for e in explanations]
        df["risk_pct"] = [e["risk_pct"] for e in explanations]
        df["alert_tier"] = [e["alert_tier"] for e in explanations]
        df["is_alert"] = [e["is_alert"] for e in explanations]
        df["reason_codes"] = [" | ".join(e["reason_codes"]) for e in explanations]

        return df


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_explainer(params: dict | None = None) -> AMLExplainer:
    """
    Load the persisted model artifact and return a ready-to-use AMLExplainer.

    Parameters
    ----------
    params : dict, optional
        Pre-loaded params dict. If None, loads from config/params.yaml.

    Returns
    -------
    AMLExplainer
    """
    if params is None:
        params = _load_params()

    model_path = ROOT / "models" / "xgboost_aml.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            "Run `python src/models/train.py` first."
        )

    artifact = joblib.load(model_path)
    return AMLExplainer(artifact, params)


# ---------------------------------------------------------------------------
# CLI: run on sample test transactions
# ---------------------------------------------------------------------------

def main() -> None:
    params = _load_params()
    explainer = load_explainer(params)

    test_path = ROOT / "data" / "processed" / "test.csv"
    if not test_path.exists():
        print("[shap_explainer] No test data found. Run the pipeline first.")
        return

    test_df = pd.read_csv(test_path)
    alerts = test_df[test_df["is_fraud"] == 1].head(5)

    print("\n" + "=" * 70)
    print("SAMPLE ALERT EXPLANATIONS (top 5 fraud cases)")
    print("=" * 70)

    for idx, row in alerts.iterrows():
        explanation = explainer.explain_single(row.drop("is_fraud"))
        print(f"\nTransaction index: {idx}")
        print(f"  Risk Score   : {explanation['risk_pct']}  [{explanation['alert_tier']}]")
        print(f"  Reason Codes :")
        for rc in explanation["reason_codes"]:
            print(f"    • {rc}")
        print(f"  Top SHAP features:")
        for tf in explanation["top_features"]:
            print(f"    {tf['feature']:<30} SHAP={tf['shap']:+.4f}")


if __name__ == "__main__":
    main()
