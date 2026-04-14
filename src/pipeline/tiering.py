"""
Tiered AML Scoring Pipeline
============================
Converts raw XGBoost fraud probabilities into a rank-ordered, tier-based
alert queue — the standard decision architecture used in production AML systems.

Scoring flow
------------
    features -> XGBoost -> fraud_score (0-1) -> global rank -> tier assignment

Tier assignment is strictly rank-based.  Probability thresholds are never used
to make decisions; the tier boundaries (critical_k, high_k, medium_k) are
business parameters configured in config/params.yaml.

    CRITICAL  top critical_k ranks          full investigator review
    HIGH      next high_k ranks             automated hold + priority queue
    MEDIUM    next medium_k ranks           soft flag / step-up authentication
    LOW       all remaining                 no action

Public API
----------
    score_transactions(df, model, feature_cols)          -> np.ndarray
    assign_tiers(df, model, feature_cols, tier_cfg)      -> pd.DataFrame
    evaluate_tiers(df_tiered, label_col)                 -> dict
    print_tier_report(metrics)                           -> None
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TIER_ORDER: list[str] = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_transactions(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
) -> np.ndarray:
    """
    Produce a fraud probability for every row in df.

    Missing feature columns are filled with 0 and a warning is raised so
    the pipeline degrades gracefully under schema drift.

    Parameters
    ----------
    df : pd.DataFrame
        Feature matrix containing at least the columns in feature_cols.
    model : XGBClassifier
        Trained XGBoost model extracted from the saved artifact dict.
    feature_cols : list[str]
        Ordered feature list used during training.

    Returns
    -------
    np.ndarray, shape (n,)
        Fraud probability per row in [0, 1].
    """
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


# ---------------------------------------------------------------------------
# Tier assignment
# ---------------------------------------------------------------------------

def assign_tiers(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    tier_cfg: dict,
    *,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Score every transaction, rank globally, and assign a priority tier.

    No probability threshold is used.  Every transaction receives a rank
    (1 = highest fraud probability) and a tier determined solely by where
    that rank falls relative to the configured bucket sizes.

    Parameters
    ----------
    df : pd.DataFrame
        Full batch to score.  May optionally include label_col.
    model : XGBClassifier
        Trained model (artifact["model"]).
    feature_cols : list[str]
        Feature columns the model expects.
    tier_cfg : dict
        Tier sizes from params.yaml > alerts > tier_k:
          {"critical_k": int, "high_k": int, "medium_k": int}
    label_col : str, optional
        Ground-truth column to preserve in output for evaluation.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with three new columns:
          fraud_score  float   model P(fraud), 6 d.p.
          rank         int     global rank in this batch (1 = highest risk)
          tier         str     CRITICAL / HIGH / MEDIUM / LOW
        Rows are sorted by rank ascending (CRITICAL first).
    """
    critical_k    = int(tier_cfg["critical_k"])
    high_cutoff   = critical_k + int(tier_cfg["high_k"])
    medium_cutoff = high_cutoff + int(tier_cfg["medium_k"])

    fraud_scores = score_transactions(df, model, feature_cols)

    # rank 1 = highest fraud_score; ties broken by original row order
    rank = (
        pd.Series(fraud_scores, index=df.index)
        .rank(method="first", ascending=False)
        .astype(int)
        .values
    )

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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_tiers(
    df_tiered: pd.DataFrame,
    label_col: str = "is_fraud",
) -> dict:
    """
    Compute tier-stratified evaluation metrics.

    Metrics produced
    ----------------
    per_tier
        Precision, alert count, and fraud count for each tier.
    combined
        CRITICAL+HIGH recall (primary safety metric), FPR, MEDIUM recall,
        and total recall across all actioned tiers.
    ranking
        AUC-ROC and AUC-PR — threshold-independent ranking quality.
    volume
        Batch-level totals for reporting.

    Parameters
    ----------
    df_tiered : pd.DataFrame
        Output of assign_tiers with label_col present.
    label_col : str
        Binary ground-truth column (0 = legitimate, 1 = fraud).

    Returns
    -------
    dict
        Nested metrics dictionary.
    """
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

    # Per-tier
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

    # CRITICAL + HIGH combined
    actionable  = np.isin(tiers, ["CRITICAL", "HIGH"])
    ch_n_alerts = int(actionable.sum())
    ch_n_fraud  = int(y_true[actionable].sum())
    ch_fp       = ch_n_alerts - ch_n_fraud

    # MEDIUM
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


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def print_tier_report(metrics: dict) -> None:
    """Print a structured tier evaluation report to stdout."""
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
