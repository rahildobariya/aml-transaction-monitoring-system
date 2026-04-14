# AML Transaction Monitoring System

A production-grade Anti-Money Laundering (AML) transaction monitoring system built on XGBoost with SHAP explainability, MLflow experiment tracking, and a Streamlit investigator dashboard.

---

## Table of Contents

- [Architecture](#architecture)
- [Alert Tiers](#alert-tiers)
- [Environment Notes](#environment-notes)
- [Dataset Information](#dataset-information)
- [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Reproducibility](#reproducibility)
- [System Limitations](#system-limitations)
- [Future Improvements](#future-improvements)
- [Project Structure](#project-structure)
- [Author](#author)

---

## Architecture

```
Raw transactions
      |
      v
src/data/load_data.py          -- load and standardise raw CSV
      |
      v
src/features/engineering.py    -- 22 behavioural + network features
      |
      v
src/models/train.py            -- XGBoost with dynamic class weighting
      |
      v
src/models/evaluate.py         -- rank-based tiered evaluation
      |
      v
app.py                         -- Streamlit investigator dashboard
```

---

## Alert Tiers

Every transaction receives a fraud score and a global rank. Tiers are assigned strictly by rank — no fixed probability threshold is used anywhere in the decision logic.

| Tier | Ranks | Operational Action |
|------|-------|--------------------|
| CRITICAL | 1 – 400 | Full investigator review |
| HIGH | 401 – 1,000 | Automated hold + priority queue |
| MEDIUM | 1,001 – 2,000 | Soft flag / step-up authentication |
| LOW | 2,001+ | No action |

Tier cutoffs are business parameters configured in `config/params.yaml` under `alerts.tier_k`.

---

## Environment Notes

- **Python version:** 3.9 or higher
- **Recommended:** use a virtual environment (`venv` or `conda`) to isolate dependencies
- **MLflow tracking** runs locally by default — no remote server required
- Ensure all paths in `config/params.yaml` are correctly set before running any pipeline step
- On Windows, activate the virtual environment with `venv\Scripts\activate` before running any commands

---

## Dataset Information

This project is built on a large-scale financial transaction dataset with approximately 6.36 million rows (configurable via `sample_size` in `config/params.yaml`).

The dataset simulates real-world financial transactions and includes the following fraudulent behaviour patterns:

- **Structuring** — transaction amounts placed just below the $10,000 reporting threshold
- **Rapid transaction bursts** — multiple high-frequency transactions within the same time window
- **New beneficiary exploitation** — first-time transfers to previously unseen receiver accounts
- **High-frequency sender behaviour** — unusually high transaction counts over 7-day rolling windows
- **Receiver aggregation** — mule accounts accumulating funds from many distinct senders
- **Multi-hop layering proxies** — behavioural signals consistent with fund layering

> **Note:** The dataset is synthetic/simulated and does not represent real production bank data. Fraud labels are ground-truth flags embedded in the simulation.

---

## Setup

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Load Data

```bash
python src/data/load_data.py
```

Reads `data/raw/transactions.csv`, standardises column names to the internal schema, and writes stratified splits to `data/processed/train.csv` and `data/processed/test.csv`.

---

### 2. Feature Engineering

```bash
python src/features/engineering.py
```

Builds 22 features across six groups:

| Group | Features |
|-------|----------|
| Temporal | `tx_hour`, `is_weekend` |
| Counterparty | `is_merchant_dest`, `is_new_beneficiary` |
| AML Flags | `structuring_flag`, `round_amount_flag`, `rapid_succession_flag`, `layering_score` |
| Velocity | `sender_tx_count_7d`, `sender_tx_count_24h`, `sender_tx_amount_7d`, `receiver_tx_count_7d`, `unique_receivers_7d` |
| Amount Deviation | `amount`, `amount_zscore`, `amount_to_avg_ratio`, `rolling_std_7d` |
| Receiver Network | `receiver_new_sender_ratio`, `receiver_inflow_concentration`, `receiver_total_inflow_7d`, `shared_counterparty_risk` |

All rolling features use time-safe past-only windows — no data leakage.

---

### 3. Train

```bash
python src/models/train.py
```

- Trains XGBoost with `scale_pos_weight` computed dynamically from the training class distribution
- Logs hyperparameters, metrics, and the model artifact to MLflow
- Saves the trained model to `models/xgboost_aml.pkl`

---

### 4. Evaluate

```bash
python src/models/evaluate.py
```

Scores the held-out test set, assigns tiers, and prints a structured report:

```
=================================================================
  AML TIERED SCORING REPORT
=================================================================

  Batch summary
  Total transactions                       40,000
  True fraud cases                          1,643
  Fraud rate                                4.11%

  Per-tier alert breakdown
  Tier          Alerts    Fraud  Precision    Coverage
  ------------ -------- -------- ---------- ----------
  CRITICAL          400      ...       ...%       ...%
  HIGH              600      ...       ...%       ...%
  MEDIUM          1,000      ...       ...%       ...%
  LOW            38,000        0      0.00%      0.00%

  Combined action metrics (CRITICAL + HIGH)
  ...
=================================================================
```

Metrics are logged to MLflow and saved to `reports/test_metrics.json`.

---

### 5. Dashboard

```bash
streamlit run app.py
```

Opens the investigator dashboard at `http://localhost:8501`.

Three views:
- **Alert Queue** — ranked transaction list with tier badges and SHAP reason codes
- **Transaction Lookup** — search by transaction ID for full risk breakdown
- **Model Performance** — per-tier precision, recall, and AUC metrics

---

### 6. MLflow UI (optional)

```bash
mlflow ui
```

Opens experiment tracker at `http://localhost:5000`.

---

## Configuration

All pipeline parameters are centralised in `config/params.yaml`:

| Section | Key | Description |
|---------|-----|-------------|
| `data` | `sample_size` | Number of rows to use (null = all 6.36M) |
| `data` | `random_seed` | Global random seed for reproducibility |
| `model.params` | `n_estimators`, `max_depth`, `learning_rate` | XGBoost hyperparameters |
| `alerts.tier_k` | `critical_k`, `high_k`, `medium_k` | Tier bucket sizes (row counts, not probability thresholds) |
| `shap` | `top_n_features` | Number of SHAP reason codes shown per alert |
| `mlflow` | `experiment_name`, `tracking_uri` | MLflow tracking configuration |

---

## Reproducibility

The following design choices ensure consistent results across runs:

- Fixed random seed defined in `config/params.yaml` and passed to all stochastic components
- Train/test split is stratified on the fraud label to preserve class distribution
- Feature engineering uses time-safe rolling windows — no future data leaks into past windows
- `scale_pos_weight` is computed deterministically from training data class counts
- OrdinalEncoder is fit exclusively on the training set and applied to validation/test sets

---

## System Limitations

- Rank-based tiering is bounded by the operational alert budget — recall is constrained by `critical_k + high_k` capacity
- The model operates on transaction-level features only; true multi-hop graph traversal (layering chains, shell company rings) is not yet implemented
- Rolling windows are computed over time steps, not real calendar time — behaviour may differ on datasets with irregular step distributions
- No concept drift detection or automatic retraining trigger in the current version
- Cold-start problem: new accounts with no transaction history receive weakly-informative velocity features

---

## Future Improvements

- **Graph Neural Network (GNN)** layer for detecting fraud rings and multi-hop laundering chains
- **Real-time streaming pipeline** using Apache Kafka and Spark Structured Streaming
- **Online learning** for adaptive model updates as fraud patterns evolve
- **Device and IP fingerprinting** features for channel-level risk signals
- **Drift detection** with Population Stability Index (PSI) to trigger automated retraining
- **LightGBM comparison experiment** — faster training with comparable AUC on large tabular datasets
- **Global + local SHAP views** in the dashboard for model-level explainability

---

## Project Structure

```
.
├── app.py                          # Streamlit investigator dashboard
├── config/
│   └── params.yaml                 # All pipeline parameters
├── data/
│   ├── raw/                        # Source CSV (gitignored)
│   └── processed/                  # Engineered feature splits (gitignored)
├── models/
│   └── xgboost_aml.pkl            # Trained model artifact (gitignored)
├── reports/                        # Evaluation output — metrics + scored CSV (gitignored)
├── requirements.txt
└── src/
    ├── data/
    │   └── load_data.py            # Data loading and schema standardisation
    ├── features/
    │   └── engineering.py          # 22-feature behavioural engineering pipeline
    ├── models/
    │   ├── train.py                # XGBoost training with MLflow logging
    │   └── evaluate.py             # Tiered evaluation and metrics reporting
    ├── pipeline/
    │   └── tiering.py              # Rank-based tier assignment and evaluation API
    └── explainability/
        └── shap_explainer.py       # SHAP TreeExplainer + AML reason codes
```

---

## Author

**Rahil Dobariya**

**Contact**: rahildobariya2024@gmail.com
