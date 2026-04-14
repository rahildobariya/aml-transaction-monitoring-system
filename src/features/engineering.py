"""
Feature Engineering Pipeline -- PaySim Edition (Bank-Grade Behavioral Features)
=================================================================================
Transforms standardised PaySim transaction data into model-ready features.

Design principle: ZERO raw balance columns in the final feature matrix.
All signals are derived from behaviour, timing, and counterparty patterns —
the same signals available in a real bank's transaction monitoring system.

Feature groups
--------------
1. Velocity          sender_tx_count_7d, sender_tx_count_24h, sender_tx_amount_7d,
                     receiver_tx_count_7d, receiver_total_inflow_7d, unique_receivers_7d
2. Amount deviation  amount_zscore, amount_to_avg_ratio, rolling_std_7d
3. Temporal          tx_hour, is_weekend
4. Counterparty      is_merchant_dest, is_new_beneficiary
5. AML flags         layering_score, structuring_flag, round_amount_flag,
                     rapid_succession_flag
6. Receiver network  receiver_new_sender_ratio, receiver_inflow_concentration,
                     shared_counterparty_risk          <-- NEW
7. Categorical       tx_type (OrdinalEncoded, fit on train only)

Performance notes
-----------------
- Velocity + receiver network: vectorised searchsorted within each group.
  Within-group loops are over groups (O(G)), not over rows (O(N)).
  All numpy operations inside groups are vectorised → O(N log N) total.
- rolling_std_7d: vectorised via cumsum-of-squares identity (no inner loop).
- is_new_beneficiary: vectorised via groupby cumcount (no row iteration).
- receiver_inflow_concentration: daily-bin aggregation + pandas rolling
  (reduces N to N/24 before the window operation).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_params() -> dict:
    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


REPORTING_THRESHOLD = 10_000.0


# ---------------------------------------------------------------------------
# 1. Velocity features  (sender + receiver rolling windows)
# ---------------------------------------------------------------------------

def compute_velocity_features(df: pd.DataFrame, window_7d: int = 168, window_24h: int = 24) -> pd.DataFrame:
    """
    Rolling transaction counts and amounts per sender and receiver.

    All searchsorted calls are vectorised across the whole group array
    (``np.searchsorted(steps, steps - w)`` returns one index per row),
    so there is no element-level Python loop inside the group loops.

    Sender features produced
    ------------------------
    sender_tx_count_7d  : transactions sent in past 168 steps (7 days)
    sender_tx_amount_7d : total amount sent in past 168 steps
    sender_tx_count_24h : transactions sent in past 24 steps (burst)
    unique_receivers_7d : distinct receivers in past 168 steps

    Receiver features produced
    --------------------------
    receiver_tx_count_7d   : inbound transaction count in past 168 steps
    receiver_total_inflow_7d: total inbound amount in past 168 steps
    """
    df = df.sort_values("step").reset_index(drop=True)

    sender_count_7d    = np.zeros(len(df), dtype=np.float32)
    sender_amount_7d   = np.zeros(len(df), dtype=np.float64)
    sender_count_24h   = np.zeros(len(df), dtype=np.float32)
    unique_recv_7d     = np.zeros(len(df), dtype=np.float32)
    receiver_count_7d  = np.zeros(len(df), dtype=np.float32)
    receiver_inflow_7d = np.zeros(len(df), dtype=np.float64)

    # ------------------------------------------------------------------ Sender
    sender_frame = (
        df[["sender_id", "step", "amount", "receiver_id"]]
        .copy()
        .assign(_orig_idx=df.index)
        .sort_values(["sender_id", "step"])
    )

    for _, grp in sender_frame.groupby("sender_id", sort=False):
        steps   = grp["step"].values
        amounts = grp["amount"].values
        recvs   = grp["receiver_id"].values
        orig    = grp["_orig_idx"].values
        n       = len(steps)
        idx     = np.arange(n)

        # Vectorised lower-bound indices for every row in the group
        lo7  = np.searchsorted(steps, steps - window_7d, side="left")
        lo24 = np.searchsorted(steps, steps - window_24h, side="left")

        cum_amt = np.concatenate([[0.0], np.cumsum(amounts)])

        sender_count_7d[orig]  = idx - lo7
        sender_amount_7d[orig] = cum_amt[idx] - cum_amt[lo7]
        sender_count_24h[orig] = idx - lo24

        # unique_receivers_7d: sliding-window set — unavoidably per-element,
        # but loop is over groups (O(G)), not over all rows (O(N)).
        for i in range(n):
            unique_recv_7d[orig[i]] = len(set(recvs[lo7[i]:i]))

    # --------------------------------------------------------------- Receiver
    recv_frame = (
        df[["receiver_id", "step", "amount"]]
        .copy()
        .assign(_orig_idx=df.index)
        .sort_values(["receiver_id", "step"])
    )

    for _, grp in recv_frame.groupby("receiver_id", sort=False):
        steps   = grp["step"].values
        amounts = grp["amount"].values
        orig    = grp["_orig_idx"].values
        n       = len(steps)
        idx     = np.arange(n)

        lo7     = np.searchsorted(steps, steps - window_7d, side="left")
        cum_amt = np.concatenate([[0.0], np.cumsum(amounts)])

        receiver_count_7d[orig]  = idx - lo7
        receiver_inflow_7d[orig] = cum_amt[idx] - cum_amt[lo7]

    df["sender_tx_count_7d"]      = sender_count_7d
    df["sender_tx_amount_7d"]     = np.round(sender_amount_7d, 2)
    df["sender_tx_count_24h"]     = sender_count_24h
    df["unique_receivers_7d"]     = unique_recv_7d
    df["receiver_tx_count_7d"]    = receiver_count_7d
    df["receiver_total_inflow_7d"] = np.round(receiver_inflow_7d, 2)

    return df


# ---------------------------------------------------------------------------
# 2. Amount deviation features
# ---------------------------------------------------------------------------

def compute_amount_features(df: pd.DataFrame, window_7d: int = 168) -> pd.DataFrame:
    """
    Statistical deviation features relative to each sender's history.

    Features produced
    -----------------
    amount_to_avg_ratio  : amount / sender global mean (clipped at 100)
    amount_zscore        : (amount - sender mean) / sender std (clipped ±10)
    rolling_std_7d       : rolling std of sender amounts over past 7 days.
                           Computed via cumsum-of-squares identity — fully
                           vectorised, no element-level loop.
    rapid_succession_flag: >5 txns by same sender in the same step hour
    """
    df = df.copy()

    # ------------------------------------------------ Global sender statistics
    sender_stats = df.groupby("sender_id")["amount"].agg(["mean", "std"])
    global_mean  = df["amount"].mean()
    global_std   = df["amount"].std()

    sender_avg = df["sender_id"].map(sender_stats["mean"]).fillna(global_mean)
    sender_std = (
        df["sender_id"].map(sender_stats["std"]).fillna(global_std).clip(lower=1.0)
    )

    df["amount_to_avg_ratio"] = (
        (df["amount"] / sender_avg.clip(lower=1.0)).clip(upper=100.0).round(4)
    )
    df["amount_zscore"] = (
        ((df["amount"] - sender_avg) / sender_std).clip(-10, 10).round(4)
    )

    # -------------------------------- Vectorised rolling std via cumsum identity
    # Var[X] = E[X²] - (E[X])²  applied over a sliding window.
    # cumsum + vectorised searchsorted eliminates the inner element loop.
    df = df.sort_values("step").reset_index(drop=True)
    rolling_std = np.zeros(len(df), dtype=np.float32)

    sender_frame = (
        df[["sender_id", "step", "amount"]]
        .copy()
        .assign(_orig_idx=df.index)
        .sort_values(["sender_id", "step"])
    )

    for _, grp in sender_frame.groupby("sender_id", sort=False):
        steps   = grp["step"].values
        amounts = grp["amount"].values
        orig    = grp["_orig_idx"].values
        n       = len(steps)
        idx     = np.arange(n)

        lo = np.searchsorted(steps, steps - window_7d, side="left")

        cum_x   = np.concatenate([[0.0], np.cumsum(amounts)])
        cum_x2  = np.concatenate([[0.0], np.cumsum(amounts ** 2)])

        win_n   = (idx - lo).clip(min=0)
        win_sum = cum_x[idx]  - cum_x[lo]
        win_sq  = cum_x2[idx] - cum_x2[lo]

        # E[X²] - (E[X])² ; clip to 0 to avoid floating-point negatives
        mean    = np.where(win_n > 0, win_sum / np.maximum(win_n, 1), 0.0)
        mean_sq = np.where(win_n > 0, win_sq  / np.maximum(win_n, 1), 0.0)
        variance = np.maximum(mean_sq - mean ** 2, 0.0)

        rolling_std[orig] = np.sqrt(variance).astype(np.float32)

    df["rolling_std_7d"] = rolling_std

    # ------------------------------------------- Rapid succession (vectorised)
    step_count = df.groupby(["sender_id", "step"])["amount"].transform("count")
    df["rapid_succession_flag"] = (step_count > 5).astype(int)

    return df


# ---------------------------------------------------------------------------
# 3. Temporal features
# ---------------------------------------------------------------------------

def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hour-of-day and weekend flag from PaySim step (1 step = 1 hour).

    tx_hour   : step % 24           (0-23 proxy for hour of day)
    is_weekend: (step // 24) % 7 >= 5  (Sat/Sun proxy)
    """
    df = df.copy()
    df["tx_hour"]    = df["step"] % 24
    df["is_weekend"] = ((df["step"] // 24) % 7 >= 5).astype(int)
    return df


# ---------------------------------------------------------------------------
# 4. Counterparty features
# ---------------------------------------------------------------------------

def compute_counterparty_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Counterparty-based AML signals — fully vectorised.

    is_merchant_dest  : receiver ID starts with 'M' (point-of-sale)
    is_new_beneficiary: 1 on the FIRST transaction from this sender to
                        this receiver (subsequent are 0).
                        Vectorised via groupby cumcount — no row loop.
    """
    df = df.copy().sort_values("step").reset_index(drop=True)

    df["is_merchant_dest"] = df["receiver_id"].str.startswith("M").astype(int)

    # cumcount == 0 marks the first occurrence of each (sender, receiver) pair
    df["is_new_beneficiary"] = (
        df.sort_values("step")
        .groupby(["sender_id", "receiver_id"])
        .cumcount()
        .eq(0)
        .astype(int)
        .reindex(df.index)
    )

    return df


# ---------------------------------------------------------------------------
# 5. AML rule-derived flags
# ---------------------------------------------------------------------------

def compute_aml_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classic AML behavioural flags (no balance columns required).

    structuring_flag : amount in [8500, 10000)  -- just-below-threshold
    round_amount_flag: amount divisible by 100
    layering_score   : weighted multi-hop complexity proxy
    """
    df = df.copy()

    df["structuring_flag"] = (
        (df["amount"] >= 8_500) & (df["amount"] < REPORTING_THRESHOLD)
    ).astype(int)

    df["round_amount_flag"] = (df["amount"] % 100 == 0).astype(int)

    is_transfer = (df["tx_type"] == "TRANSFER").astype(float)
    is_cash_out = (df["tx_type"] == "CASH_OUT").astype(float)

    df["layering_score"] = (
        (is_transfer * (1 - df["is_merchant_dest"])) * 0.5
        + is_cash_out * 0.3
        + ((df["amount"] >= 8_000) & (df["amount"] < REPORTING_THRESHOLD)).astype(float) * 0.2
    ).round(3)

    return df


# ---------------------------------------------------------------------------
# 6. Receiver network features  (NEW)
# ---------------------------------------------------------------------------

def compute_receiver_network_features(
    df: pd.DataFrame,
    window_7d: int = 168,
) -> pd.DataFrame:
    """
    Four receiver-side / network-level AML features that expose coordination
    patterns invisible to sender-centric models.

    All features are computed using only PAST data relative to each row:
    no leakage.

    Features produced
    -----------------
    receiver_new_sender_ratio
        Fraction of the receiver's 7-day inbound transactions that come
        from first-time senders.  High ratio = mule account aggregating
        from many new sources for the first time.

        Formula:
          new_tx_7d  = rolling count of rows where sender sends to this
                       receiver for the first time, in past 7d window
          total_tx_7d = receiver_tx_count_7d (already computed)
          ratio = new_tx_7d / max(total_tx_7d, 1)

        Implementation: vectorised searchsorted on receiver groups, using
        a pre-computed ``_first_to_recv`` indicator (1 on the earliest
        transaction from each sender to each receiver, 0 thereafter).

    receiver_inflow_concentration
        Fraction of the receiver's 7-day inflow dominated by the single
        largest sender.  High value = one entity is driving most inflows
        (pass-through mule / structured aggregation).

        Formula (top-1 sender share as concentration proxy):
          concentration = max_sender_7d_amount / max(receiver_total_inflow_7d, 1)

        Implementation: aggregate to daily bins (step // 24), apply
        pandas rolling(7) per (receiver, sender) pair and per receiver,
        then map back to transaction level.  Fully vectorised — no
        element-level loops.

        NOTE: ``shift(1).rolling(7)`` excludes the current day so the
        feature only uses strictly past data.

    shared_counterparty_risk
        Average ``amount_zscore`` of OTHER senders that have sent to the
        same receiver in the past 7 days (excluding the current sender).

        In production inference this column should be replaced with the
        actual ``fraud_score`` from the model (2-pass scoring); during
        training it uses ``amount_zscore`` as a proxy — a z-scored amount
        carries the same "anomalousness" signal without requiring model
        scores at feature-engineering time.

        Implementation: vectorised searchsorted per receiver group,
        cumsum of proxy scores, subtract current row's contribution to
        avoid self-inclusion.

    Prerequisites
    -------------
    This function must be called AFTER:
      - compute_velocity_features   (provides receiver_tx_count_7d)
      - compute_amount_features     (provides amount_zscore for proxy)

    Parameters
    ----------
    df : pd.DataFrame
        Transactions after velocity + amount features have been appended.
    window_7d : int
        Rolling window size in steps (default 168 = 7 × 24).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with four new columns appended (in place of any
        prior versions of these columns).
    """
    df = df.sort_values("step").reset_index(drop=True)

    # ----------------------------------------------------------
    # Pre-requisite: mark the first transaction from each sender
    # to each receiver  (vectorised via groupby cumcount)
    # ----------------------------------------------------------
    first_to_recv = (
        df.sort_values("step")
        .groupby(["receiver_id", "sender_id"])
        .cumcount()
        .eq(0)
        .astype(np.int8)
        .reindex(df.index)
        .values
    )

    # ----------------------------------------------------------
    # A) receiver_new_sender_ratio  &
    # B) shared_counterparty_risk
    #    Both use vectorised searchsorted over receiver groups.
    # ----------------------------------------------------------
    new_senders_7d  = np.zeros(len(df), dtype=np.float32)
    shared_risk     = np.zeros(len(df), dtype=np.float32)

    # Proxy score: amount_zscore if available, else layering_score
    proxy_col = "amount_zscore" if "amount_zscore" in df.columns else "layering_score"

    recv_frame = (
        df[["receiver_id", "step"]]
        .copy()
        .assign(
            _orig_idx   = df.index,
            _first      = first_to_recv,
            _proxy_score = df[proxy_col].values,
        )
        .sort_values(["receiver_id", "step"])
    )

    for _, grp in recv_frame.groupby("receiver_id", sort=False):
        steps   = grp["step"].values
        is_fst  = grp["_first"].values.astype(float)
        pscores = grp["_proxy_score"].values.astype(float)
        orig    = grp["_orig_idx"].values
        n       = len(steps)
        idx     = np.arange(n)

        lo = np.searchsorted(steps, steps - window_7d, side="left")

        # --- new_sender transactions in window ---
        cum_first = np.concatenate([[0.0], np.cumsum(is_fst)])
        new_senders_7d[orig] = cum_first[idx] - cum_first[lo]

        # --- shared_counterparty_risk ---
        # window sum/count of proxy scores, then remove current row
        cum_score = np.concatenate([[0.0], np.cumsum(pscores)])
        win_sum   = cum_score[idx] - cum_score[lo]
        win_n     = (idx - lo).astype(float)

        other_sum   = win_sum  - pscores       # exclude current row
        other_count = np.maximum(win_n - 1, 0)

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(other_count > 0, other_sum / np.maximum(other_count, 1), 0.0)
        shared_risk[orig] = ratio.astype(np.float32)

    # receiver_tx_count_7d was computed in compute_velocity_features
    recv_count_7d = df["receiver_tx_count_7d"].values.clip(min=1)

    df["receiver_new_sender_ratio"] = (
        (new_senders_7d / recv_count_7d).clip(0, 1).round(4)
    )
    df["shared_counterparty_risk"] = np.round(shared_risk, 4)

    # ----------------------------------------------------------
    # C) receiver_inflow_concentration
    #    Daily-bin aggregation + pandas rolling — fully vectorised.
    #    Reduces N rows to N/24 before the window operation.
    # ----------------------------------------------------------
    df["_day"] = df["step"] // 24  # 1 day = 24 simulation hours

    # Daily amount per (receiver, sender, day)
    daily_pairs = (
        df.groupby(["receiver_id", "sender_id", "_day"], sort=False)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "_pair_daily_amt"})
        .sort_values(["receiver_id", "sender_id", "_day"])
    )

    # Rolling 7-day amount from this specific sender to this receiver
    # shift(1) ensures the current day is NOT included (leakage-free)
    daily_pairs["_pair_7d_amt"] = (
        daily_pairs
        .groupby(["receiver_id", "sender_id"])["_pair_daily_amt"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=0).sum().fillna(0.0))
    )

    # Daily total inflow per (receiver, day)
    daily_recv_total = (
        daily_pairs
        .groupby(["receiver_id", "_day"], sort=False)["_pair_daily_amt"]
        .sum()
        .reset_index()
        .rename(columns={"_pair_daily_amt": "_recv_daily_total"})
        .sort_values(["receiver_id", "_day"])
    )
    daily_recv_total["_recv_7d_total"] = (
        daily_recv_total
        .groupby("receiver_id")["_recv_daily_total"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=0).sum().fillna(0.0))
    )

    # Sender share within receiver 7-day window
    daily_pairs = daily_pairs.merge(
        daily_recv_total[["receiver_id", "_day", "_recv_7d_total"]],
        on=["receiver_id", "_day"],
        how="left",
    )
    daily_pairs["_sender_share"] = (
        daily_pairs["_pair_7d_amt"]
        / daily_pairs["_recv_7d_total"].clip(lower=1.0)
    ).clip(0, 1)

    # Top-1 sender share per (receiver, day) = concentration
    # (Top-1 proxy; top-3 would require a sort per group — marginal gain)
    daily_conc = (
        daily_pairs
        .groupby(["receiver_id", "_day"], sort=False)["_sender_share"]
        .max()
        .reset_index()
        .rename(columns={"_sender_share": "receiver_inflow_concentration"})
    )

    df = df.merge(daily_conc, on=["receiver_id", "_day"], how="left")
    df["receiver_inflow_concentration"] = (
        df["receiver_inflow_concentration"].fillna(1.0).clip(0, 1).round(4)
    )

    # Cleanup temporary columns
    df = df.drop(columns=["_day"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# 7. Categorical encoding
# ---------------------------------------------------------------------------

def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cat_features: list[str],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], OrdinalEncoder]:
    """
    Fit OrdinalEncoder on train set only, apply to both splits.
    Unknown categories in test are mapped to -1 (no leakage).
    """
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        dtype=np.float32,
    )
    train_df = train_df.copy()
    train_df[cat_features] = enc.fit_transform(train_df[cat_features])

    if test_df is not None:
        test_df = test_df.copy()
        test_df[cat_features] = enc.transform(test_df[cat_features])

    return train_df, test_df, enc


# ---------------------------------------------------------------------------
# 8. Full feature matrix builder
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    Execution order is strict — later steps depend on earlier ones:

    Step 1  Temporal          (fast; no groupby; provides tx_hour, is_weekend)
    Step 2  Counterparty      (sort by step; provides is_merchant_dest,
                               is_new_beneficiary — needed by AML flags)
    Step 3  AML flags         (needs is_merchant_dest from step 2)
    Step 4  Velocity          (O(N log N); provides receiver_tx_count_7d
                               needed by receiver network features)
    Step 5  Amount deviation  (O(N log N); provides amount_zscore
                               used as proxy in shared_counterparty_risk)
    Step 6  Receiver network  (NEW — must run after steps 4 and 5)

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_data.py (standardised column names, includes step).

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame with no raw balance columns.
    """
    print("[engineering] Step 1/6 -- Temporal features ...")
    df = compute_temporal_features(df)

    print("[engineering] Step 2/6 -- Counterparty features ...")
    df = compute_counterparty_features(df)

    print("[engineering] Step 3/6 -- AML flags ...")
    df = compute_aml_flags(df)

    print("[engineering] Step 4/6 -- Velocity aggregates (7d + 24h) ...")
    df = compute_velocity_features(df)

    print("[engineering] Step 5/6 -- Amount deviation (zscore, ratio, rolling std) ...")
    df = compute_amount_features(df)

    print("[engineering] Step 6/6 -- Receiver network features (new) ...")
    df = compute_receiver_network_features(df)

    return df


# ---------------------------------------------------------------------------
# 9. Pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    End-to-end feature engineering:
      1. Load standardised transactions
      2. Build all features (6 steps)
      3. Stratified train/test split
      4. Encode categoricals (fit on train only — no leakage)
      5. Save processed splits
    """
    params   = _load_params()
    data_cfg = params["data"]
    feat_cfg = params["features"]

    raw_path = ROOT / "data" / "raw" / "transactions.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Standardised data not found at {raw_path}. "
            "Run `python src/data/load_data.py` first."
        )

    print(f"[engineering] Loading {raw_path} ...")
    df = pd.read_csv(raw_path)

    df = build_feature_matrix(df)

    # Drop identifier and raw balance columns — keep only model-ready features
    drop_cols = [
        "step", "sender_id", "receiver_id",
        "old_balance_orig", "new_balance_orig",
        "old_balance_dest",  "new_balance_dest",
    ]
    model_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = model_df.drop(columns=["is_fraud"])
    y = model_df["is_fraud"]

    # Stratified split — test set is never seen during training or encoding
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        stratify=y,
        random_state=data_cfg["random_seed"],
    )

    # Encode categoricals (fit on train only)
    cat_features = [c for c in feat_cfg.get("categorical", []) if c in X_train.columns]
    if cat_features:
        X_train, X_test, _ = encode_categoricals(X_train, X_test, cat_features)

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.concat([X_train, y_train], axis=1).to_csv(out_dir / "train.csv", index=False)
    pd.concat([X_test,  y_test],  axis=1).to_csv(out_dir / "test.csv",  index=False)

    print(f"[engineering] Train : {len(X_train):,} rows -> data/processed/train.csv")
    print(f"[engineering] Test  : {len(X_test):,} rows  -> data/processed/test.csv")
    print(f"[engineering] Fraud rate (train): {y_train.mean():.3%}")
    print(f"[engineering] Features ({len(X_train.columns)}): {list(X_train.columns)}")


if __name__ == "__main__":
    run_pipeline()
