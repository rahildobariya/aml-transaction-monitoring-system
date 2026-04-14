"""
PaySim Kaggle Dataset Loader
==============================
Reads the real PaySim CSV (paysim_raw.csv), standardises column names to the
pipeline convention, and writes transactions.csv to data/raw/.

PaySim schema → pipeline mapping
---------------------------------
step            → step           (time proxy, 1 step = 1 hour)
type            → tx_type
amount          → amount
nameOrig        → sender_id
oldbalanceOrg   → old_balance_orig
newbalanceOrig  → new_balance_orig
nameDest        → receiver_id
oldbalanceDest  → old_balance_dest
newbalanceDest  → new_balance_dest
isFraud         → is_fraud
isFlaggedFraud  → (dropped)

Scale strategy
--------------
The full dataset has 6.36M rows with ~0.13% fraud. SMOTE on this volume will
OOM on most laptops. By default we keep ALL fraud rows + randomly sample legit
rows so the total is data.sample_size (default 200,000). Set sample_size to null
in params.yaml to use the full dataset.

Usage:
    python src/data/load_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _load_params() -> dict:
    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


COLUMN_RENAME = {
    "step": "step",             # time proxy (1 step = 1 hour) -- needed for velocity features
    "type": "tx_type",
    "amount": "amount",
    "nameOrig": "sender_id",
    "oldbalanceOrg": "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest": "receiver_id",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud": "is_fraud",
}


def load_and_standardise(raw_path: Path, params: dict) -> pd.DataFrame:
    """
    Load the raw PaySim CSV, rename columns, drop unused columns, and
    optionally downsample to a manageable size.

    Parameters
    ----------
    raw_path : Path
        Path to paysim_raw.csv
    params : dict
        Loaded from config/params.yaml

    Returns
    -------
    pd.DataFrame
        Cleaned, standardised DataFrame.
    """
    data_cfg = params["data"]
    sample_size = data_cfg.get("sample_size", 200_000)
    seed = data_cfg.get("random_seed", 42)

    print(f"[load_data] Reading {raw_path} …")
    df = pd.read_csv(raw_path, usecols=list(COLUMN_RENAME.keys()))
    df = df.rename(columns=COLUMN_RENAME)

    print(f"[load_data] Full dataset  : {len(df):,} rows")
    print(f"[load_data] Fraud count   : {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.3%})")

    # Downsample: keep all fraud + random sample of legit
    if sample_size is not None and len(df) > sample_size:
        fraud_df = df[df["is_fraud"] == 1]
        legit_df = df[df["is_fraud"] == 0]

        n_legit = sample_size - len(fraud_df)
        if n_legit <= 0:
            # Edge case: more fraud rows than sample_size — just use fraud
            df = fraud_df
        else:
            legit_sample = legit_df.sample(n=min(n_legit, len(legit_df)), random_state=seed)
            df = pd.concat([fraud_df, legit_sample], ignore_index=True)

        # Shuffle
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"[load_data] After sampling: {len(df):,} rows "
              f"(fraud rate: {df['is_fraud'].mean():.3%})")

    return df


def main() -> None:
    params = _load_params()

    raw_path = ROOT / "data" / "raw" / "paysim_raw.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {raw_path}. "
            "Download it from Kaggle and place it at data/raw/paysim_raw.csv"
        )

    df = load_and_standardise(raw_path, params)

    out_path = ROOT / "data" / "raw" / "transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"[load_data] Saved -> {out_path}")
    print(f"[load_data] Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
