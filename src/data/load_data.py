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


# maps raw PaySim column names to the internal names used throughout the pipeline
COLUMN_RENAME = {
    "step":           "step",             # 1 step = 1 hour, needed for rolling windows
    "type":           "tx_type",
    "amount":         "amount",
    "nameOrig":       "sender_id",
    "oldbalanceOrg":  "old_balance_orig",
    "newbalanceOrig": "new_balance_orig",
    "nameDest":       "receiver_id",
    "oldbalanceDest": "old_balance_dest",
    "newbalanceDest": "new_balance_dest",
    "isFraud":        "is_fraud",
}


def load_and_standardise(raw_path: Path, params: dict) -> pd.DataFrame:
    data_cfg    = params["data"]
    sample_size = data_cfg.get("sample_size", 200_000)
    seed        = data_cfg.get("random_seed", 42)

    print(f"[load_data] Reading {raw_path} ...")
    df = pd.read_csv(raw_path, usecols=list(COLUMN_RENAME.keys()))
    df = df.rename(columns=COLUMN_RENAME)

    print(f"[load_data] Full dataset  : {len(df):,} rows")
    print(f"[load_data] Fraud count   : {df['is_fraud'].sum():,} ({df['is_fraud'].mean():.3%})")

    # keep all fraud rows and randomly sample legit rows to hit the target size
    # this avoids OOM issues with SMOTE on 6M rows
    if sample_size is not None and len(df) > sample_size:
        fraud_df = df[df["is_fraud"] == 1]
        legit_df = df[df["is_fraud"] == 0]

        n_legit = sample_size - len(fraud_df)
        if n_legit <= 0:
            df = fraud_df
        else:
            legit_sample = legit_df.sample(n=min(n_legit, len(legit_df)), random_state=seed)
            df = pd.concat([fraud_df, legit_sample], ignore_index=True)

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
