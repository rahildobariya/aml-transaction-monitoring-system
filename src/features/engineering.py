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


# $10k is the FinCEN reporting threshold — amounts just below it are a structuring signal
REPORTING_THRESHOLD = 10_000.0


def compute_velocity_features(df: pd.DataFrame, window_7d: int = 168, window_24h: int = 24) -> pd.DataFrame:
    df = df.sort_values("step").reset_index(drop=True)

    sender_count_7d    = np.zeros(len(df), dtype=np.float32)
    sender_amount_7d   = np.zeros(len(df), dtype=np.float64)
    sender_count_24h   = np.zeros(len(df), dtype=np.float32)
    unique_recv_7d     = np.zeros(len(df), dtype=np.float32)
    receiver_count_7d  = np.zeros(len(df), dtype=np.float32)
    receiver_inflow_7d = np.zeros(len(df), dtype=np.float64)

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

        # searchsorted returns a lower-bound index for each row — fully vectorised
        lo7  = np.searchsorted(steps, steps - window_7d,  side="left")
        lo24 = np.searchsorted(steps, steps - window_24h, side="left")

        cum_amt = np.concatenate([[0.0], np.cumsum(amounts)])

        sender_count_7d[orig]  = idx - lo7
        sender_amount_7d[orig] = cum_amt[idx] - cum_amt[lo7]
        sender_count_24h[orig] = idx - lo24

        # unique receivers can't be done with cumsum, but loop is over groups not rows
        for i in range(n):
            unique_recv_7d[orig[i]] = len(set(recvs[lo7[i]:i]))

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

    df["sender_tx_count_7d"]       = sender_count_7d
    df["sender_tx_amount_7d"]      = np.round(sender_amount_7d, 2)
    df["sender_tx_count_24h"]      = sender_count_24h
    df["unique_receivers_7d"]      = unique_recv_7d
    df["receiver_tx_count_7d"]     = receiver_count_7d
    df["receiver_total_inflow_7d"] = np.round(receiver_inflow_7d, 2)

    return df


def compute_amount_features(df: pd.DataFrame, window_7d: int = 168) -> pd.DataFrame:
    df = df.copy()

    sender_stats = df.groupby("sender_id")["amount"].agg(["mean", "std"])
    global_mean  = df["amount"].mean()
    global_std   = df["amount"].std()

    sender_avg = df["sender_id"].map(sender_stats["mean"]).fillna(global_mean)
    sender_std = df["sender_id"].map(sender_stats["std"]).fillna(global_std).clip(lower=1.0)

    df["amount_to_avg_ratio"] = (df["amount"] / sender_avg.clip(lower=1.0)).clip(upper=100.0).round(4)
    df["amount_zscore"]       = ((df["amount"] - sender_avg) / sender_std).clip(-10, 10).round(4)

    # rolling std via Var[X] = E[X²] - E[X]² — avoids a nested loop
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

        lo     = np.searchsorted(steps, steps - window_7d, side="left")
        cum_x  = np.concatenate([[0.0], np.cumsum(amounts)])
        cum_x2 = np.concatenate([[0.0], np.cumsum(amounts ** 2)])

        win_n   = (idx - lo).clip(min=0)
        win_sum = cum_x[idx]  - cum_x[lo]
        win_sq  = cum_x2[idx] - cum_x2[lo]

        mean     = np.where(win_n > 0, win_sum / np.maximum(win_n, 1), 0.0)
        mean_sq  = np.where(win_n > 0, win_sq  / np.maximum(win_n, 1), 0.0)
        variance = np.maximum(mean_sq - mean ** 2, 0.0)  # clip to 0 for float precision issues

        rolling_std[orig] = np.sqrt(variance).astype(np.float32)

    df["rolling_std_7d"] = rolling_std

    step_count = df.groupby(["sender_id", "step"])["amount"].transform("count")
    df["rapid_succession_flag"] = (step_count > 5).astype(int)

    return df


def compute_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # PaySim uses steps where 1 step = 1 hour, so step % 24 gives hour of day
    df["tx_hour"]    = df["step"] % 24
    df["is_weekend"] = ((df["step"] // 24) % 7 >= 5).astype(int)
    return df


def compute_counterparty_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("step").reset_index(drop=True)

    df["is_merchant_dest"] = df["receiver_id"].str.startswith("M").astype(int)

    # cumcount == 0 means this is the first time this sender has sent to this receiver
    df["is_new_beneficiary"] = (
        df.sort_values("step")
        .groupby(["sender_id", "receiver_id"])
        .cumcount()
        .eq(0)
        .astype(int)
        .reindex(df.index)
    )

    return df


def compute_aml_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # structuring: deliberately keeping amount just below the reporting threshold
    df["structuring_flag"]  = ((df["amount"] >= 8_500) & (df["amount"] < REPORTING_THRESHOLD)).astype(int)
    df["round_amount_flag"] = (df["amount"] % 100 == 0).astype(int)

    is_transfer = (df["tx_type"] == "TRANSFER").astype(float)
    is_cash_out = (df["tx_type"] == "CASH_OUT").astype(float)

    df["layering_score"] = (
        (is_transfer * (1 - df["is_merchant_dest"])) * 0.5
        + is_cash_out * 0.3
        + ((df["amount"] >= 8_000) & (df["amount"] < REPORTING_THRESHOLD)).astype(float) * 0.2
    ).round(3)

    return df


def compute_receiver_network_features(df: pd.DataFrame, window_7d: int = 168) -> pd.DataFrame:
    df = df.sort_values("step").reset_index(drop=True)

    # mark the first time each sender sends to each receiver — used for new_sender_ratio
    first_to_recv = (
        df.sort_values("step")
        .groupby(["receiver_id", "sender_id"])
        .cumcount()
        .eq(0)
        .astype(np.int8)
        .reindex(df.index)
        .values
    )

    new_senders_7d = np.zeros(len(df), dtype=np.float32)
    shared_risk    = np.zeros(len(df), dtype=np.float32)

    # use amount_zscore as a proxy for fraud risk during training
    # in production you'd use the actual model score from a first pass
    proxy_col = "amount_zscore" if "amount_zscore" in df.columns else "layering_score"

    recv_frame = (
        df[["receiver_id", "step"]]
        .copy()
        .assign(
            _orig_idx    = df.index,
            _first       = first_to_recv,
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

        cum_first = np.concatenate([[0.0], np.cumsum(is_fst)])
        new_senders_7d[orig] = cum_first[idx] - cum_first[lo]

        cum_score = np.concatenate([[0.0], np.cumsum(pscores)])
        win_sum   = cum_score[idx] - cum_score[lo]
        win_n     = (idx - lo).astype(float)

        # exclude current row so we're measuring other senders, not this one
        other_sum   = win_sum  - pscores
        other_count = np.maximum(win_n - 1, 0)

        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(other_count > 0, other_sum / np.maximum(other_count, 1), 0.0)
        shared_risk[orig] = ratio.astype(np.float32)

    recv_count_7d = df["receiver_tx_count_7d"].values.clip(min=1)
    df["receiver_new_sender_ratio"] = (new_senders_7d / recv_count_7d).clip(0, 1).round(4)
    df["shared_counterparty_risk"]  = np.round(shared_risk, 4)

    # aggregate to daily bins before rolling — reduces N rows to N/24
    df["_day"] = df["step"] // 24

    daily_pairs = (
        df.groupby(["receiver_id", "sender_id", "_day"], sort=False)["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "_pair_daily_amt"})
        .sort_values(["receiver_id", "sender_id", "_day"])
    )

    # shift(1) makes sure the current day is excluded from the window (no leakage)
    daily_pairs["_pair_7d_amt"] = (
        daily_pairs
        .groupby(["receiver_id", "sender_id"])["_pair_daily_amt"]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=0).sum().fillna(0.0))
    )

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

    daily_pairs = daily_pairs.merge(
        daily_recv_total[["receiver_id", "_day", "_recv_7d_total"]],
        on=["receiver_id", "_day"],
        how="left",
    )
    daily_pairs["_sender_share"] = (
        daily_pairs["_pair_7d_amt"] / daily_pairs["_recv_7d_total"].clip(lower=1.0)
    ).clip(0, 1)

    # top-1 sender share = concentration (how much one sender dominates this receiver)
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

    df = df.drop(columns=["_day"], errors="ignore")
    return df


def encode_categoricals(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    cat_features: list[str],
) -> tuple[pd.DataFrame, Optional[pd.DataFrame], OrdinalEncoder]:
    # fit only on train set — applying to test separately avoids data leakage
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


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # order matters here — later steps depend on columns from earlier ones
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

    # receiver network needs receiver_tx_count_7d (step 4) and amount_zscore (step 5)
    print("[engineering] Step 6/6 -- Receiver network features ...")
    df = compute_receiver_network_features(df)

    return df


def run_pipeline() -> None:
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

    # drop IDs and balance columns — model should only see behaviour, not account balances
    drop_cols = [
        "step", "sender_id", "receiver_id",
        "old_balance_orig", "new_balance_orig",
        "old_balance_dest", "new_balance_dest",
    ]
    model_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    X = model_df.drop(columns=["is_fraud"])
    y = model_df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_cfg["test_size"],
        stratify=y,
        random_state=data_cfg["random_seed"],
    )

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
