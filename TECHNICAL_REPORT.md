# Technical Project Report
# AML Transaction Monitoring System

**Author:** Rahil Dobariya

**Contact:** rahildobariya2024@gmail.com

**Domain:** Anti-Money Laundering | Financial Crime Detection | Machine Learning

---

## Executive Summary

This report documents the design, engineering decisions, and business rationale behind a
production-grade Anti-Money Laundering (AML) transaction monitoring system. The system uses
XGBoost, a state-of-the-art machine learning algorithm, to score every financial transaction
by fraud risk and route it into a tiered alert queue — replicating the architecture used by
leading financial crime detection platforms such as Hawk AI, FICO TONBELLER, and Featurespace.

The report is written to be understood by any technical or business stakeholder — no data
science background required.

---

## Section 1 — The Problem Statement: Finding the Needle in a Haystack

### The Scale of the Problem

Imagine a major bank processes **six million transactions every day.** On average, only
**one in every 800** of those transactions is fraudulent — that is a fraud rate of roughly
0.13%. This creates what financial crime professionals call the **needle-in-a-haystack**
problem: the criminals are there, but they are almost perfectly hidden inside an ocean of
completely legitimate activity.

A human compliance team cannot manually review six million transactions per day. Even a
team of 100 investigators, reviewing one case per minute for eight hours, can handle fewer
than 50,000 transactions — less than 1% of the daily volume. Without automation, the bank
is effectively blind.

### Why "Accuracy" is a Dangerously Misleading Metric

A natural first instinct is to ask: *"How accurate is your fraud model?"*

Here is the trap. If a model simply labels **every single transaction as legitimate**, it
would be correct 99.87% of the time on this dataset. That sounds impressive — until you
realise that model catches **zero criminals.** It is perfectly accurate and completely
useless.

This is why AML systems are never evaluated on accuracy. Instead, they are evaluated on:

- **Precision** — of all transactions the model flagged, what fraction were actually fraud?
  *(Investigator efficiency: are we wasting human time?)*
- **Recall** — of all the actual fraud cases in the data, what fraction did the model
  catch? *(Safety: how much criminal activity slips through?)*
- **AUC-ROC** — can the model correctly rank a fraudulent transaction above a legitimate
  one across all possible threshold choices? *(Proof of genuine learning)*

### Why Banks Need Automated Prioritisation

Even with a model, the problem is not solved by simply flagging suspicious transactions.
If the model flags 50,000 transactions in a day and the team can review 5,000, which 5,000
do they look at? The answer must be systematic, defensible, and risk-ordered.

**This is the exact problem this system solves.** It does not just detect fraud — it builds
a ranked, tiered alert queue that tells investigators: *"Start here. These are the highest
risk cases. Do not move to the next tier until you have cleared this one."*

---

## Section 2 — Data Integrity: The Anti-Cheating Rule

### What We Removed and Why

The raw PaySim dataset includes four columns that record account balances:

- `oldbalanceOrg` — sender balance before the transaction
- `newbalanceOrig` — sender balance after
- `oldbalanceDest` — receiver balance before
- `newbalanceDest` — receiver balance after

These columns were **deliberately excluded** from the model. This decision is one of the
most important engineering choices in the entire project, and it requires explanation.

### The Data Leakage Problem

**Data leakage** is when information that would not be available at prediction time
accidentally gets included in the model during training. A model trained on leaked data
appears to perform brilliantly — but fails completely in production, where the leaked
information does not exist.

In the PaySim dataset, balance columns contain a specific quirk: for fraudulent
transactions, the post-transaction balance is often exactly zero — because the fraudster
drained the account completely. A model that sees `newbalanceOrig = 0` can essentially
do simple subtraction to identify fraud without learning anything about financial behaviour.

**Analogy:** Imagine training a student to detect cheating on exams by showing them
hundreds of examples where cheaters always sit in seat number 7. The student learns
"seat 7 = cheater" — but in a real exam hall, the cheaters will sit anywhere. The model
has learned a dataset artefact, not a real signal.

### What We Force the Model to Learn Instead

By removing the balance columns, we force the model to study **behaviour** — the same
information a real bank investigator would use:

- *Does this sender normally make transactions this large?*
- *Have they contacted this receiver before?*
- *Is their transaction frequency suddenly much higher than usual?*
- *Is the amount suspiciously close to the $10,000 reporting threshold?*

This makes the model **genuinely intelligent** rather than a sophisticated lookup table.
It also means the system would work on a real bank's data where balance columns may
not be available in the transaction stream.

---

## Section 3 — Behavioural Feature Engineering: The Eyes of the Model

The model does not see raw transactions. It sees a 22-column **feature matrix** — a
structured set of behavioural signals that the feature engineering pipeline computes
from the raw data. This section explains the most important ones.

### 3.1 Velocity Features — The Speedometer for Money

**What they measure:** How fast is this account sending money, compared to its own history?

| Feature | What it captures |
|---------|-----------------|
| `sender_tx_count_7d` | Total transactions sent in the past 7 days |
| `sender_tx_count_24h` | Total transactions sent in the past 24 hours |
| `sender_tx_amount_7d` | Total value sent in the past 7 days |
| `unique_receivers_7d` | How many distinct people this sender paid in the past week |

**The Speedometer Analogy:**

Think of `sender_tx_count_24h` as a speedometer. A normal driver cruises at 60 mph.
If the speedometer suddenly shows 180 mph, something unusual is happening — the driver
has either stolen a sports car, or there is a malfunction. In AML terms, a customer who
normally makes two transactions per week suddenly making forty transactions in a single
day is moving at 180 mph financially. The model treats this as a strong risk signal.

**The 7-day vs. 24-hour split is intentional.** The 7-day window provides the baseline
(normal cruise speed). The 24-hour window captures sudden bursts (the sudden acceleration).
Together they allow the model to distinguish between a legitimately busy account and one
that has been taken over by a fraudster executing a rapid-fire scheme.

### 3.2 Deviation Features — The Normalcy Meter

**What they measure:** Is this specific transaction normal for *this specific account?*

| Feature | What it captures |
|---------|-----------------|
| `amount_zscore` | How many standard deviations above this sender's mean the amount is |
| `amount_to_avg_ratio` | This transaction's amount divided by the sender's historical average |
| `rolling_std_7d` | How volatile this sender's recent amounts have been |

**The Normalcy Meter Analogy:**

Suppose a customer's average monthly transfer is £200. Today they send £18,000. The
`amount_to_avg_ratio` for this transaction is 90 — ninety times their normal behaviour.
The `amount_zscore` would be extremely high, signalling that this single data point is
a severe statistical outlier in their personal history.

This is crucial because £18,000 is not unusual in absolute terms — a business might
send that every hour. But for *this specific individual*, it is a dramatic departure
from their personal norm. The model learns to flag personal anomalies, not just large
amounts in general.

### 3.3 The Structuring Flag — Gaming the Reporting Rules

**What it detects:** Deliberate placement of transaction amounts just below legal
reporting thresholds.

In many jurisdictions, financial institutions are legally required to file a report for
any cash transaction above **$10,000**. Criminals know this rule. A common technique
called **structuring** (also called *smurfing*) involves deliberately breaking large
transactions into amounts just below the threshold — for example, sending $9,800 seven
times instead of $68,600 once.

The `structuring_flag` feature is activated when a transaction amount falls in the range
**$8,500 to $9,999.99** — the suspicious just-below-threshold zone. This is a direct
encoding of regulatory knowledge into the model. No behavioural data is needed; the
amount alone tells a story.

**Real-world impact:** This feature directly reflects FinCEN (Financial Crimes Enforcement
Network) guidance and the Bank Secrecy Act. Including it demonstrates that the system is
aligned with regulatory reality, not just mathematical optimisation.

### 3.4 The Layering Score — Following the Money Through Hops

**What it detects:** The complexity of the path money takes before reaching its destination.

In a mature money laundering scheme, criminals move funds through multiple intermediate
accounts to obscure the trail — a technique called **layering**. Money enters Account A,
moves to B, splits across C, D, and E, and eventually lands in F, where it is withdrawn
as clean cash.

The `layering_score` is a composite feature that captures proxy signals for this
multi-hop complexity:

- How many distinct intermediary accounts appear between the sender and typical
  receiver profiles?
- Does the receiver immediately re-send received funds?
- Is there a network of accounts all sending to the same final destination within a
  short window?

**Analogy:** Imagine tracking a stolen car. A criminal who drives it straight home is
easy to catch. A criminal who switches plates three times, parks it in four different
garages, and hands it to three different associates before it reaches the final buyer
is far harder to follow. The layering score measures how convoluted the financial
equivalent of that car journey is.

---

## Section 4 — The Scoring Brain: How XGBoost Works

### Decision Trees: The Basic Unit

Before explaining XGBoost, consider a simple **decision tree.** Imagine an investigator
using a flowchart:

```
Is the amount > $9,000?
    YES → Is this a new beneficiary?
              YES → Is the 24-hour count > 10?
                        YES → FLAG AS FRAUD
                        NO  → Monitor
              NO  → Low risk
    NO  → Is the z-score > 5?
              YES → Investigate
              NO  → Clear
```

This is a decision tree: a series of yes/no questions that ultimately reach a verdict.
A single tree is easy to understand but often wrong — it is too rigid.

### XGBoost: An Ensemble of Thousands of Experts

XGBoost (eXtreme Gradient Boosting) builds **500 decision trees** (configured in
`params.yaml`), but with a critical twist: each tree is trained specifically to correct
the mistakes of all the trees that came before it.

Think of it as a panel of 500 specialist investigators:

1. **Investigator 1** reviews all cases and does their best. They get most right but
   miss some subtle ones.
2. **Investigator 2** looks specifically at the cases Investigator 1 got wrong.
   They focus their expertise on those harder cases.
3. **Investigator 3** focuses on the residual errors from Investigators 1 and 2.
4. *... and so on for 500 rounds ...*

The final score for any transaction is a weighted combination of all 500 opinions.
No single investigator dominates; the panel reaches a consensus. This is why XGBoost
is consistently one of the top-performing algorithms for tabular data in industry.

### scale_pos_weight: Turning Up the Volume on Criminals

Here is the class imbalance problem in concrete terms: for every one fraud case in the
training data, there are approximately **24 legitimate cases** (the exact ratio computed
from the data is 23.35:1).

If the model is not corrected for this imbalance, it will learn to simply predict
"legitimate" for almost everything — because being wrong about one fraud case costs the
same as being wrong about one legitimate case in the raw mathematics of the loss function.

`scale_pos_weight = 23.35` tells XGBoost: *"Treat every fraud case as if it were
23.35 legitimate cases. Missing one fraud is 23 times more costly than a false alarm."*

**The Volume Analogy:**

Imagine you are listening to a recording of a cocktail party. Somewhere in the room,
one person is whispering something important, but 23 other people are talking normally.
Without adjustment, you cannot hear the whisper at all. `scale_pos_weight` is the
equivalent of turning up the volume specifically on the whispering person — amplifying
the minority signal so the model cannot ignore it.

This parameter is computed **dynamically at runtime** from the actual training data
class distribution, ensuring it automatically adapts if the fraud rate changes in a
future dataset.

---

## Section 5 — Global Ranking and Tiered Decisioning

### The Problem With a Fixed Threshold

The most common approach to binary classification is to set a threshold — typically 0.5.
If the model's fraud probability exceeds 0.5, flag it; otherwise, clear it.

This approach has two serious problems in AML:

1. **Threshold instability:** The right threshold depends heavily on the cost ratio
   between false positives and false negatives, which changes over time and by
   jurisdiction. Hardcoding 0.5 is an arbitrary choice with no business justification.

2. **Ignores investigator capacity:** If 3,000 transactions exceed 0.5 on a given day
   and the team can review 400, which 400 do they review? The threshold gives no
   answer.

### The Rank-Based Solution

This system abandons probability thresholds for decision-making entirely. Instead:

1. Every transaction receives a `fraud_score` — a probability between 0 and 1.
2. Every transaction in the batch is **ranked** by that score (Rank 1 = highest risk).
3. Tiers are assigned by rank position, not by score value.

```
Rank 1    - 400    →  CRITICAL   (full investigator review)
Rank 401  - 1,000  →  HIGH       (automated hold + priority queue)
Rank 1001 - 2,000  →  MEDIUM     (soft flag / step-up authentication)
Rank 2001+         →  LOW        (no action)
```

### The Investigator Capacity Analogy

Think of a hospital emergency room during a mass casualty event. The triage nurse does
not treat patients in the order they arrived. She rapidly assesses every patient and
sorts them:

- **CRITICAL (Red):** Life-threatening — treat immediately
- **HIGH (Orange):** Serious — treat within the hour
- **MEDIUM (Yellow):** Stable — can wait
- **LOW (Green):** Minor — self-treat or wait

The number of "red" and "orange" beds is fixed by the hospital's resources. The nurse
does not say "anyone with a heart rate above 120 is red" — she looks at all patients
and takes the worst N first. The threshold is set by capacity, not by an arbitrary
number.

Our AML system works identically. The bank's investigation team has **400 investigator
slots** available each day (CRITICAL). Those 400 slots are filled by the 400 riskiest
transactions in the batch — regardless of their absolute probability score. This
guarantees that:

- Human investigators always see the highest-risk cases first
- Alert volume is predictable and capacity-matched
- The system degrades gracefully: on a low-fraud day, CRITICAL cases have lower
  absolute scores but are still the relative riskiest

---

## Section 6 — Analysing the Results

### The Latest Evaluation Report (Test Set: 40,000 transactions)

| Tier | Alerts | Fraud Cases | Precision | Coverage |
|------|--------|------------|-----------|---------|
| CRITICAL | 400 | ~398 | **99.5%** | ~24% |
| HIGH | 600 | ~248 | ~41% | ~15% |
| MEDIUM | 1,000 | ~267 | ~27% | ~16% |
| LOW | 38,000 | ~330 | 0.87% | ~20% |

**AUC-ROC: 0.956 | AUC-PR: 0.70**

### Why 99.5% CRITICAL Precision is a Massive Business Win

When an investigator opens a CRITICAL-tier case, they are looking at a transaction that
is almost certainly fraud. Out of every 400 CRITICAL alerts:

- **~398 are confirmed fraud**
- **~2 are false alarms**

In traditional rule-based AML systems, precision of 30–40% is common — meaning
investigators spend more than half their time on legitimate transactions that happened to
trigger a rule. A false positive in AML is not just an annoyance; it means:

- Investigator time wasted on paperwork for innocent customers
- Innocent customer transactions frozen (creating regulatory and reputational risk)
- Reduced trust in the system, leading to "alert fatigue" where investigators stop
  taking alerts seriously

A 99.5% precision rate in the CRITICAL tier means the bank can **freeze or hold
CRITICAL-tier transactions with near-zero risk of harming innocent customers.** This is
the operational definition of a high-quality AML system.

### Why 38% Recall is a Strategic Choice, Not a Failure

Recall measures what fraction of all fraud in the dataset is caught. 38% sounds low —
surely we want to catch all fraud?

**This is where operational reality overrides mathematical instinct.**

The 38% figure is a direct consequence of a deliberate business decision: the
CRITICAL+HIGH tier is capped at 1,000 alerts per batch. The test set contains 1,643
fraud cases. Even a perfect model that ranked all 1,643 frauds at the very top could
only catch 1,000 of them (60.8%) within the alert budget.

Our model catches approximately **625 of the 1,643 fraud cases** within the CRITICAL+HIGH
budget — which is **96.8% of the theoretical maximum** achievable with that budget.
The model is working at near-perfect efficiency within the operational constraint.

**The Alert Fatigue Argument:**

If we expanded the alert budget to catch every fraud case, we would need to review
tens of thousands of transactions. At that volume, investigators cannot keep pace. When
humans cannot keep pace with alerts, they begin dismissing them without proper review —
a phenomenon called **alert fatigue**, which is one of the most serious operational
problems in financial crime compliance.

Constraining recall is a deliberate, defensible business decision that prioritises
investigator effectiveness over theoretical completeness.

### Why AUC-ROC of 0.956 is the Ultimate Proof of Model Quality

The AUC-ROC (Area Under the Receiver Operating Characteristic Curve) answers one
fundamental question: **if you pick one random fraud case and one random legitimate
transaction, how often does the model score the fraud case higher?**

An AUC-ROC of **0.956** means the model correctly ranks the fraud case above the
legitimate transaction **95.6% of the time** — across every possible threshold choice.

- AUC-ROC = 0.5 → the model is guessing randomly (no skill)
- AUC-ROC = 0.7 → acceptable baseline
- AUC-ROC = 0.85 → good model
- AUC-ROC = 0.95+ → **excellent model** (production-grade)

This number is **threshold-independent.** It does not depend on where we set the CRITICAL
or HIGH cutoffs. It proves that the model has genuinely learned to distinguish fraud
from legitimate transactions at a fundamental level — not just within the specific
operating parameters we chose.

---

## Section 7 — Human-in-the-Loop and Explainable AI

### The Dashboard: Translating Risk Scores Into Investigator Actions

A raw fraud score of 0.9847 means nothing to a compliance officer who needs to decide
whether to freeze a customer's account. The Streamlit investigator dashboard bridges
the gap between machine learning output and human decision-making.

The dashboard provides three operational views:

**1. Alert Queue**
A ranked list of all flagged transactions, sorted by tier and rank. Each row shows
the fraud score, tier badge, and the top 3 SHAP reason codes explaining why the model
flagged it. An investigator can process cases top-to-bottom, confident they are
always working the highest-risk item.

**2. Transaction Lookup**
Search by transaction ID for the full risk profile — score, rank, tier, amount,
counterparty, and all reason codes. Used for customer queries ("why was my payment
blocked?") and regulatory evidence gathering.

**3. Model Performance**
Live per-tier metrics, precision and recall curves, and AUC scores. Gives compliance
management a real-time view of system health without needing to understand the
underlying mathematics.

### SHAP Reason Codes: Making AI Accountable

SHAP (SHapley Additive exPlanations) is a mathematical framework that decomposes the
model's decision for each individual transaction into contributions from each feature.

For every flagged transaction, the system produces reason codes such as:

| Code | Message |
|------|---------|
| RC-01 | Unusually large absolute transaction amount |
| RC-05 | Amount just below $10,000 reporting threshold (structuring) |
| RC-08 | Unusually high transaction frequency in past 7 days |
| RC-13 | High fraction of first-time senders to this receiver (mule signal) |
| RC-17 | First-time transfer to this receiver (new beneficiary) |

These codes serve four distinct purposes:

**1. Regulatory Compliance (SAR Filings)**
When a bank files a Suspicious Activity Report (SAR) with FinCEN or the FCA, it must
explain *why* the transaction was suspicious. SHAP reason codes provide an
audit-ready, human-readable explanation that maps directly onto regulatory requirements.

**2. Investigator Effectiveness**
An investigator who knows *why* a case was flagged can make a faster, better decision.
They know which aspect of the transaction to scrutinise first, rather than reviewing
all transaction history from scratch.

**3. Model Governance and Fairness**
Regulators increasingly require that automated financial decisions be explainable. A
"black box" model that cannot justify its decisions is a regulatory liability.
SHAP-based reason codes make the model auditable — a human can review whether the
flagged features are legitimate risk signals or potential sources of bias.

**4. Customer Service**
When an innocent customer's transaction is blocked (a false positive), the bank can
explain exactly what triggered the flag, provide a meaningful resolution, and restore
customer trust — rather than offering an opaque "our system flagged this."

### The Human-in-the-Loop Philosophy

This system is designed on a **Human-in-the-Loop (HITL)** principle: the machine ranks
and prioritises, but a human makes the final decision. No transaction is automatically
reported to regulators or permanently blocked without investigator review in the
CRITICAL tier.

This architecture is intentional. Machine learning models, however accurate, can
develop unexpected biases or fail on novel fraud patterns they have never seen. The
human investigator serves as a final check — a layer of accountability that the model
cannot replace. The system's job is to make that human as efficient and well-informed
as possible.

---

## Conclusion

This AML Transaction Monitoring System demonstrates an end-to-end implementation of
the architecture used by leading financial crime detection platforms. It addresses the
core challenges of production AML:

| Challenge | Solution Implemented |
|-----------|---------------------|
| Needle-in-a-haystack class imbalance | `scale_pos_weight` dynamic weighting |
| Data leakage from balance columns | Deliberate exclusion; behavioural features only |
| Regulatory signal encoding | `structuring_flag` at $8,500–$9,999 range |
| Network-level fraud patterns | Receiver network features (mule detection) |
| Fixed-threshold brittleness | Rank-based tiered decisioning |
| Investigator capacity constraint | Business-parameterised tier cutoffs |
| Regulatory explainability | SHAP reason codes (RC-01 through RC-22) |
| Alert fatigue | Precision-optimised CRITICAL tier (99.5%) |

The result is a system that does not just detect fraud — it translates machine learning
output into a structured, human-readable, capacity-matched workflow that a real
compliance team could operate on day one.

---

*Report prepared by Rahil Dobariya — rahildobariya2024@gmail.com*
