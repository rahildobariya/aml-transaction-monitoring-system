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


class AMLExplainer:
    def __init__(self, artifact: dict, params: dict) -> None:
        self.model        = artifact["model"]
        self.threshold    = artifact["threshold"]
        self.feature_names = artifact["feature_names"]

        shap_cfg = params["shap"]
        self.top_n           = shap_cfg["top_n_features"]
        self.reason_code_map = shap_cfg["reason_codes"]

        self.explainer = shap.TreeExplainer(self.model)

    def explain_batch(self, X: pd.DataFrame) -> list[dict[str, Any]]:
        X           = X.reindex(columns=self.feature_names, fill_value=0)
        shap_values = self.explainer.shap_values(X)
        probs       = self.model.predict_proba(X)[:, 1]

        return [self._build_explanation(prob, shap_values[i]) for i, prob in enumerate(probs)]

    def explain_single(self, x: pd.Series) -> dict[str, Any]:
        return self.explain_batch(pd.DataFrame([x]))[0]

    def _build_explanation(self, prob: float, shap_row: np.ndarray) -> dict[str, Any]:
        alert_tier = self._get_alert_tier(prob)
        is_alert   = prob >= self.threshold

        # sort features by how much they contributed to the prediction
        feature_shap = sorted(
            zip(self.feature_names, shap_row),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        top_features = []
        reason_codes = []

        for feat_name, shap_val in feature_shap:
            if len(reason_codes) >= self.top_n:
                break
            # only include features that push the score higher, not ones that lower it
            if shap_val > 0:
                top_features.append({
                    "feature":   feat_name,
                    "shap":      round(float(shap_val), 4),
                    "direction": "increases",
                })
                rc = self.reason_code_map.get(feat_name)
                if rc:
                    reason_codes.append(rc)

        # fallback if everything has negative SHAP (unlikely but possible)
        if not reason_codes:
            for feat_name, shap_val in feature_shap[:self.top_n]:
                rc = self.reason_code_map.get(feat_name, f"RC-UNK: {feat_name} is anomalous")
                reason_codes.append(rc)

        return {
            "risk_score":   round(float(prob), 4),
            "risk_pct":     f"{prob * 100:.1f}%",
            "alert_tier":   alert_tier,
            "is_alert":     bool(is_alert),
            "reason_codes": reason_codes,
            "shap_values":  {f: round(float(v), 4) for f, v in zip(self.feature_names, shap_row)},
            "top_features": top_features,
        }

    @staticmethod
    def _get_alert_tier(prob: float) -> str:
        # these thresholds are only used for display in the dashboard
        # actual tier assignment uses rank-based logic in tiering.py
        if prob >= 0.80:
            return "CRITICAL"
        elif prob >= 0.60:
            return "HIGH"
        elif prob >= 0.30:
            return "MEDIUM"
        else:
            return "LOW"

    def annotate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [c for c in self.feature_names if c in df.columns]
        explanations = self.explain_batch(df[feature_cols])

        df = df.copy()
        df["risk_score"]   = [e["risk_score"]              for e in explanations]
        df["risk_pct"]     = [e["risk_pct"]                for e in explanations]
        df["alert_tier"]   = [e["alert_tier"]              for e in explanations]
        df["is_alert"]     = [e["is_alert"]                for e in explanations]
        df["reason_codes"] = [" | ".join(e["reason_codes"]) for e in explanations]

        return df


def load_explainer(params: dict | None = None) -> AMLExplainer:
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


def main() -> None:
    params    = _load_params()
    explainer = load_explainer(params)

    test_path = ROOT / "data" / "processed" / "test.csv"
    if not test_path.exists():
        print("[shap_explainer] No test data found. Run the pipeline first.")
        return

    test_df = pd.read_csv(test_path)
    alerts  = test_df[test_df["is_fraud"] == 1].head(5)

    print("\n" + "=" * 70)
    print("SAMPLE ALERT EXPLANATIONS (top 5 fraud cases)")
    print("=" * 70)

    for idx, row in alerts.iterrows():
        explanation = explainer.explain_single(row.drop("is_fraud"))
        print(f"\nTransaction index: {idx}")
        print(f"  Risk Score   : {explanation['risk_pct']}  [{explanation['alert_tier']}]")
        print(f"  Reason Codes :")
        for rc in explanation["reason_codes"]:
            print(f"    - {rc}")
        print(f"  Top SHAP features:")
        for tf in explanation["top_features"]:
            print(f"    {tf['feature']:<30} SHAP={tf['shap']:+.4f}")


if __name__ == "__main__":
    main()
