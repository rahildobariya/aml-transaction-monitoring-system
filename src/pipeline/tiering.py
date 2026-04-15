import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TIER_ORDER: list[str] = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


def score_transactions(df: pd.DataFrame, model, feature_cols: list[str]) -> np.ndarray:
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]

    if missing:
        import warnings
        warnings.warn(
            f"{len(missing)} feature(s) missing from input — filled with 0: {missing}",
            RuntimeWarning,
            stacklevel=2,
        )
        fill_df = pd.DataFrame(0.0, index=df.index, columns=missing)
        X = pd.concat([df[available], fill_df], axis=1)[feature_cols]
    else:
        X = df[feature_cols]

    return model.predict_proba(X)[:, 1]


def assign_tiers(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    tier_cfg: dict,
    *,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    critical_k    = int(tier_cfg["critical_k"])
    high_cutoff   = critical_k + int(tier_cfg["high_k"])
    medium_cutoff = high_cutoff + int(tier_cfg["medium_k"])

    fraud_scores = score_transactions(df, model, feature_cols)

    # rank 1 = highest fraud score, ties broken by original row order
    rank = (
        pd.Series(fraud_scores, index=df.index)
        .rank(method="first", ascending=False)
        .astype(int)
        .values
    )

    # assign tier based on rank position, not probability value
    tier = np.select(
        condlist=[rank <= critical_k, rank <= high_cutoff, rank <= medium_cutoff],
        choicelist=["CRITICAL", "HIGH", "MEDIUM"],
        default="LOW",
    )

    out = df.copy()
    out["fraud_score"] = np.round(fraud_scores, 6)
    out["rank"]        = rank
    out["tier"]        = tier

    return out.sort_values("rank").reset_index(drop=True)


def evaluate_tiers(df_tiered: pd.DataFrame, label_col: str = "is_fraud") -> dict:
    if label_col not in df_tiered.columns:
        raise ValueError(
            f"Label column '{label_col}' not found.  "
            "Pass label_col=None to skip evaluation."
        )

    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true  = df_tiered[label_col].values
    y_score = df_tiered["fraud_score"].values
    tiers   = df_tiered["tier"].values

    total_fraud = int(y_true.sum())
    n_total     = len(y_true)
    total_legit = n_total - total_fraud

    per_tier: dict[str, dict] = {}
    for t in TIER_ORDER:
        mask     = tiers == t
        n_alerts = int(mask.sum())
        n_fraud  = int(y_true[mask].sum())
        per_tier[t] = {
            "n_alerts":  n_alerts,
            "n_fraud":   n_fraud,
            "precision": round(n_fraud / max(n_alerts, 1), 4),
        }

    actionable  = np.isin(tiers, ["CRITICAL", "HIGH"])
    ch_n_alerts = int(actionable.sum())
    ch_n_fraud  = int(y_true[actionable].sum())
    ch_fp       = ch_n_alerts - ch_n_fraud

    med_mask    = tiers == "MEDIUM"
    med_n_fraud = int(y_true[med_mask].sum())

    try:
        auc_roc = round(float(roc_auc_score(y_true, y_score)), 4)
        auc_pr  = round(float(average_precision_score(y_true, y_score)), 4)
    except ValueError:
        auc_roc = auc_pr = float("nan")

    return {
        "per_tier": per_tier,
        "combined": {
            "critical_high_recall":   round(ch_n_fraud / max(total_fraud, 1), 4),
            "critical_high_n_alerts": ch_n_alerts,
            "critical_high_n_fraud":  ch_n_fraud,
            "critical_high_fp":       ch_fp,
            "critical_high_fpr":      round(ch_fp / max(total_legit, 1), 4),
            "medium_recall":          round(med_n_fraud / max(total_fraud, 1), 4),
            "total_recall":           round((ch_n_fraud + med_n_fraud) / max(total_fraud, 1), 4),
        },
        "ranking": {"auc_roc": auc_roc, "auc_pr": auc_pr},
        "volume": {
            "total_transactions": n_total,
            "total_fraud":        total_fraud,
            "fraud_rate":         round(total_fraud / max(n_total, 1), 4),
        },
    }


def print_tier_report(metrics: dict) -> None:
    vol      = metrics["volume"]
    comb     = metrics["combined"]
    ranking  = metrics["ranking"]
    per_tier = metrics["per_tier"]
    sep      = "=" * 65

    print(f"\n{sep}")
    print("  AML TIERED SCORING REPORT")
    print(sep)

    print(f"\n  Batch summary")
    print(f"  {'Total transactions':<35} {vol['total_transactions']:>10,}")
    print(f"  {'True fraud cases':<35} {vol['total_fraud']:>10,}")
    print(f"  {'Fraud rate':<35} {vol['fraud_rate']:>10.2%}")

    print(f"\n  Per-tier alert breakdown")
    print(f"  {'Tier':<12} {'Alerts':>8} {'Fraud':>8} {'Precision':>10} {'Coverage':>10}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    for t in TIER_ORDER:
        if t not in per_tier:
            continue
        pt       = per_tier[t]
        coverage = pt["n_fraud"] / max(vol["total_fraud"], 1)
        print(f"  {t:<12} {pt['n_alerts']:>8,} {pt['n_fraud']:>8,} "
              f"{pt['precision']:>10.2%} {coverage:>10.2%}")

    print(f"\n  Combined action metrics (CRITICAL + HIGH)")
    print(f"  {'Alerts':<35} {comb['critical_high_n_alerts']:>10,}")
    print(f"  {'Fraud caught':<35} {comb['critical_high_n_fraud']:>10,}")
    print(f"  {'Recall':<35} {comb['critical_high_recall']:>10.2%}")
    print(f"  {'False positives':<35} {comb['critical_high_fp']:>10,}")
    print(f"  {'False positive rate':<35} {comb['critical_high_fpr']:>10.4f}")
    print(f"  {'MEDIUM tier recall':<35} {comb['medium_recall']:>10.2%}")
    print(f"  {'Total recall (all tiers)':<35} {comb['total_recall']:>10.2%}")

    print(f"\n  Ranking quality (threshold-independent)")
    print(f"  {'AUC-ROC':<35} {ranking['auc_roc']:>10.4f}")
    print(f"  {'AUC-PR':<35} {ranking['auc_pr']:>10.4f}")
    print(f"\n{sep}\n")
