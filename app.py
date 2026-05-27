import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.explainability.shap_explainer import AMLExplainer, load_explainer
from src.pipeline.tiering import assign_tiers, TIER_ORDER

st.set_page_config(
    page_title="AML Transaction Monitoring",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# tier colours used across the whole dashboard
st.markdown(
    """
    <style>
    .tier-CRITICAL { color: #FF2D55; font-weight: bold; }
    .tier-HIGH     { color: #FF6B35; font-weight: bold; }
    .tier-MEDIUM   { color: #FFD60A; font-weight: bold; }
    .tier-LOW      { color: #30D158; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

TIER_COLORS = {
    "CRITICAL": "#FF2D55",
    "HIGH":     "#FF6B35",
    "MEDIUM":   "#FFD60A",
    "LOW":      "#30D158",
}


def _load_params() -> dict:
    with open(ROOT / "config" / "params.yaml") as f:
        return yaml.safe_load(f)


# cache the explainer so it doesn't reload on every interaction
@st.cache_resource(show_spinner="Loading model ...")
def _get_explainer() -> AMLExplainer:
    return load_explainer()


@st.cache_resource(show_spinner="Loading model artifact ...")
def _get_artifact() -> dict:
    path = ROOT / "models" / "xgboost_aml.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `python src/models/train.py` first."
        )
    return joblib.load(path)


@st.cache_data(show_spinner="Loading test data ...")
def _load_test_data() -> pd.DataFrame:
    path = ROOT / "data" / "processed" / "test.csv"
    if path.exists():
        return pd.read_csv(path)
    demo_path = ROOT / "demo" / "sample_data.csv"
    if demo_path.exists():
        return pd.read_csv(demo_path)
    return pd.DataFrame()


# df_bytes is used as the cache key since streamlit can't hash DataFrames directly
@st.cache_data(show_spinner="Scoring and ranking transactions ...", ttl=3600)
def _score_and_rank(_artifact: dict, df_bytes: bytes) -> pd.DataFrame:
    import io
    df = pd.read_parquet(io.BytesIO(df_bytes))

    params       = _load_params()
    tier_cfg     = params["alerts"]["tier_k"]
    feat_cfg     = params["features"]
    feature_cols = [
        c for c in (feat_cfg["numeric"] + feat_cfg.get("categorical", []))
        if c in df.columns
    ]

    tiered = assign_tiers(
        df,
        model=_artifact["model"],
        feature_cols=feature_cols,
        tier_cfg=tier_cfg,
        label_col="is_fraud" if "is_fraud" in df.columns else None,
    )

    tiered = tiered.rename(columns={"fraud_score": "risk_score", "tier": "alert_tier"})
    tiered["risk_pct"] = (tiered["risk_score"] * 100).round(1).astype(str) + "%"
    tiered["is_alert"] = tiered["alert_tier"].isin(["CRITICAL", "HIGH"])
    return tiered


def _get_scored(artifact: dict, raw_df: pd.DataFrame, n: int = 2000) -> pd.DataFrame:
    import io
    buf = io.BytesIO()
    raw_df.head(n).to_parquet(buf, index=False)
    return _score_and_rank(artifact, buf.getvalue())


def _risk_gauge(score: float, tier: str) -> go.Figure:
    color = TIER_COLORS.get(tier, "#888")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": color},
            # background segments go green -> yellow -> orange -> red
            "steps": [
                {"range": [0, 30],   "color": "#1E3A2F"},
                {"range": [30, 60],  "color": "#3A3010"},
                {"range": [60, 80],  "color": "#3A1E10"},
                {"range": [80, 100], "color": "#3A0A15"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": score * 100,
            },
        },
        title={"text": f"Fraud Score -- {tier}", "font": {"size": 18}},
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="white",
    )
    return fig


def _shap_waterfall(explanation: dict, max_features: int = 10) -> go.Figure:
    # sort by absolute SHAP value so the biggest drivers show first
    top    = sorted(explanation["shap_values"].items(),
                    key=lambda x: abs(x[1]), reverse=True)[:max_features]
    feats  = [f[0] for f in top]
    values = [f[1] for f in top]
    fig = go.Figure(go.Bar(
        x=values, y=feats, orientation="h",
        marker_color=["#FF2D55" if v > 0 else "#30D158" for v in values],
        text=[f"{v:+.4f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="SHAP Feature Impact  (red = increases fraud risk)",
        xaxis_title="SHAP Value",
        yaxis={"autorange": "reversed"},
        height=max(300, len(feats) * 32 + 80),
        margin=dict(l=210, r=80, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="white",
        xaxis=dict(gridcolor="#333"),
    )
    return fig


def _tier_badge(tier: str) -> str:
    color = TIER_COLORS.get(tier, "#888")
    return f'<span style="color:{color};font-weight:bold">{tier}</span>'


def render_sidebar() -> tuple[str, list[str]]:
    st.sidebar.title("AML Monitor")
    st.sidebar.markdown("---")
    view = st.sidebar.radio(
        "View",
        ["Alert Queue", "Transaction Lookup", "Model Performance"],
    )
    st.sidebar.markdown("---")
    tiers = st.sidebar.multiselect(
        "Alert Tiers",
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Rank-based tiering  |  No probability thresholds")
    return view, tiers


def _render_drilldown(row: pd.Series, explainer: AMLExplainer) -> None:
    feature_cols = [c for c in explainer.feature_names if c in row.index]
    explanation  = explainer.explain_single(row[feature_cols])
    tier         = row.get("alert_tier", explanation["alert_tier"])
    score        = float(row.get("risk_score", explanation["risk_score"]))

    col_g, col_r = st.columns([1, 2])
    with col_g:
        st.plotly_chart(_risk_gauge(score, tier), use_container_width=True)
        st.caption(f"Global rank in batch: **#{int(row['rank'])}**")
    with col_r:
        st.markdown(f"### Reason Codes -- {_tier_badge(tier)}", unsafe_allow_html=True)
        for i, rc in enumerate(explanation["reason_codes"], 1):
            st.markdown(f"**{i}.** {rc}")
        if "amount" in row:
            st.markdown(f"**Amount:** ${row['amount']:,.2f}")
        if "tx_type" in row:
            st.markdown(f"**Type:** {row.get('tx_type', 'N/A')}")
        if "is_fraud" in row:
            label = "Legitimate" if int(row["is_fraud"]) == 0 else "Confirmed Fraud"
            st.markdown(f"**Ground Truth:** {label}")

    st.plotly_chart(_shap_waterfall(explanation), use_container_width=True)

    with st.expander("Raw Feature Values"):
        rows = [
            {"Feature": f,
             "Value":   round(float(row[f]), 4) if pd.notna(row.get(f)) else "N/A",
             "SHAP":    round(explanation["shap_values"].get(f, 0), 4)}
            for f in feature_cols
        ]
        feat_df = pd.DataFrame(rows).sort_values("SHAP", ascending=False, key=abs)
        st.dataframe(feat_df, use_container_width=True)


def render_alert_queue(
    raw_df: pd.DataFrame,
    artifact: dict,
    explainer: AMLExplainer,
    tiers: list[str],
) -> None:
    st.title("Alert Queue")

    # show demo banner when running on Streamlit Cloud with sample data
    full_path = ROOT / "data" / "processed" / "test.csv"
    demo_path = ROOT / "demo" / "sample_data.csv"
    if not full_path.exists() and demo_path.exists():
        st.info(
            "ℹ️ **Demo mode** — showing a 2,000-row sample of the synthetic dataset. "
            "Clone the repo and run the full pipeline for all 40,000 transactions."
        )

    if raw_df.empty:
        st.warning(
            "No processed data found. Run the full pipeline first:\n\n"
            "```bash\npython src/data/load_data.py\n"
            "python src/features/engineering.py\n"
            "python src/models/train.py\n```"
        )
        return

    params   = _load_params()
    tier_cfg = params["alerts"]["tier_k"]
    scored   = _get_scored(artifact, raw_df)

    # summary counts at the top
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scored", f"{len(scored):,}")
    c2.metric("CRITICAL", f"{(scored['alert_tier'] == 'CRITICAL').sum():,}",
              help=f"Top {tier_cfg['critical_k']} by fraud score")
    c3.metric("HIGH",     f"{(scored['alert_tier'] == 'HIGH').sum():,}",
              help=f"Next {tier_cfg['high_k']}")
    c4.metric("MEDIUM",   f"{(scored['alert_tier'] == 'MEDIUM').sum():,}",
              help=f"Next {tier_cfg['medium_k']}")
    c5.metric("Action required", f"{scored['is_alert'].sum():,}",
              help="CRITICAL + HIGH")

    st.markdown("---")

    filtered = (
        scored[scored["alert_tier"].isin(tiers)]
        .sort_values("rank")
        .reset_index(drop=True)
    )
    st.subheader(f"{len(filtered):,} alerts  --  Tiers: {', '.join(tiers)}")

    if filtered.empty:
        st.info("No alerts match the selected tiers.")
        return

    # only show columns that actually exist in this batch
    display_cols = [c for c in [
        "rank", "risk_pct", "alert_tier", "amount", "tx_type",
        "structuring_flag", "is_new_beneficiary",
        "receiver_new_sender_ratio", "is_fraud",
    ] if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True, height=420)

    st.markdown("---")
    st.subheader("Transaction Drilldown")
    row_idx = st.number_input(
        "Row number to inspect",
        min_value=0, max_value=max(len(filtered) - 1, 0), value=0,
    )
    _render_drilldown(filtered.iloc[row_idx], explainer)


def render_lookup(raw_df: pd.DataFrame, explainer: AMLExplainer) -> None:
    st.title("Transaction Lookup")

    if raw_df.empty:
        st.warning("No data loaded. Run the pipeline first.")
        return

    feature_cols = explainer.feature_names
    # pre-fill form with first row values so it's not all zeros
    defaults = raw_df[[c for c in feature_cols if c in raw_df.columns]].iloc[0].to_dict()

    with st.form("lookup"):
        cols = st.columns(3)
        inputs: dict = {}
        for i, feat in enumerate(feature_cols):
            inputs[feat] = cols[i % 3].number_input(
                feat, value=float(defaults.get(feat, 0)), format="%.4f"
            )
        submitted = st.form_submit_button("Score Transaction")

    if submitted:
        x           = pd.Series(inputs)
        explanation = explainer.explain_single(x)
        tier        = explanation["alert_tier"]
        col1, col2  = st.columns([1, 2])
        with col1:
            st.plotly_chart(_risk_gauge(explanation["risk_score"], tier),
                            use_container_width=True)
        with col2:
            st.markdown(f"### {_tier_badge(tier)}", unsafe_allow_html=True)
            for i, rc in enumerate(explanation["reason_codes"], 1):
                st.markdown(f"**{i}.** {rc}")
        st.plotly_chart(_shap_waterfall(explanation), use_container_width=True)


def render_performance() -> None:
    st.title("Model Performance")

    metrics_path     = ROOT / "reports" / "test_metrics.json"
    val_metrics_path = ROOT / "reports" / "val_metrics.json"
    demo_metrics_path = ROOT / "demo" / "metrics.json"

    is_demo = False
    if not metrics_path.exists():
        if demo_metrics_path.exists():
            metrics_path = demo_metrics_path
            is_demo = True
        else:
            st.warning(
                "No metrics found. Run:\n\n```bash\npython src/models/evaluate.py\n```"
            )
            return

    if is_demo:
        st.info(
            "ℹ️ **Demo mode** — metrics from the full 40,000-transaction evaluation run "
            "(200,000-row training sample, AUC-ROC 0.9562). "
            "Clone the repo and run `python src/models/evaluate.py` to regenerate."
        )

    with open(metrics_path) as f:
        tm = json.load(f)
    val_metrics = {}
    if val_metrics_path.exists():
        with open(val_metrics_path) as f:
            val_metrics = json.load(f)

    comb    = tm.get("combined", {})
    ranking = tm.get("ranking",  {})
    volume  = tm.get("volume",   {})

    st.subheader("Combined Action Metrics  (CRITICAL + HIGH)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CRITICAL+HIGH Recall",  f"{comb.get('critical_high_recall', 0):.2%}")
    c2.metric("Total Recall",          f"{comb.get('total_recall', 0):.2%}")
    c3.metric("AUC-ROC",               f"{ranking.get('auc_roc', 0):.4f}")
    c4.metric("AUC-PR",                f"{ranking.get('auc_pr', 0):.4f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("False Positives",    f"{comb.get('critical_high_fp', 0):,}")
    c6.metric("FP Rate",            f"{comb.get('critical_high_fpr', 0):.4f}")
    c7.metric("Total Transactions", f"{volume.get('total_transactions', 0):,}")
    c8.metric("Total Fraud",        f"{volume.get('total_fraud', 0):,}")

    st.markdown("---")
    st.subheader("Per-Tier Breakdown")

    per_tier    = tm.get("per_tier", {})
    total_fraud = volume.get("total_fraud", 1)

    tier_rows = [
        {
            "Tier":         t,
            "Alerts":       per_tier[t]["n_alerts"],
            "Fraud Caught": per_tier[t]["n_fraud"],
            "Precision":    f"{per_tier[t]['precision']:.2%}",
            "Coverage":     f"{per_tier[t]['n_fraud'] / max(total_fraud, 1):.2%}",
        }
        for t in TIER_ORDER if t in per_tier
    ]
    if tier_rows:
        st.dataframe(pd.DataFrame(tier_rows), use_container_width=True, hide_index=True)

    col_p, col_c = st.columns(2)
    tiers_present = [t for t in TIER_ORDER if t in per_tier]

    with col_p:
        vals = [per_tier[t]["precision"] * 100 for t in tiers_present]
        fig  = go.Figure(go.Bar(
            x=tiers_present, y=vals,
            marker_color=[TIER_COLORS[t] for t in tiers_present],
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
        ))
        fig.update_layout(
            title="Precision per Tier (%)", yaxis=dict(range=[0, 108]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=320, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_c:
        vals = [per_tier[t]["n_fraud"] / max(total_fraud, 1) * 100
                for t in tiers_present]
        fig  = go.Figure(go.Bar(
            x=tiers_present, y=vals,
            marker_color=[TIER_COLORS[t] for t in tiers_present],
            text=[f"{v:.1f}%" for v in vals], textposition="outside",
        ))
        fig.update_layout(
            title="Fraud Coverage per Tier (% of total fraud)",
            yaxis=dict(range=[0, max(vals, default=50) * 1.2]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="white", height=320, margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

    # show val metrics only if the training run saved them
    if val_metrics:
        st.markdown("---")
        st.subheader("Validation Metrics (training run)")
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Val Precision", f"{val_metrics.get('val_precision', 0):.4f}")
        vc2.metric("Val Recall",    f"{val_metrics.get('val_recall',    0):.4f}")
        vc3.metric("Val AUC-ROC",   f"{val_metrics.get('val_auc_roc',   0):.4f}")
        vc4.metric("Val AUC-PR",    f"{val_metrics.get('val_auc_pr',    0):.4f}")

    st.caption("All metrics on the held-out test set. Tiers are rank-based.")


def main() -> None:
    view, tiers = render_sidebar()

    try:
        explainer = _get_explainer()
        artifact  = _get_artifact()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    raw_df = _load_test_data()

    if view == "Alert Queue":
        render_alert_queue(raw_df, artifact, explainer, tiers)
    elif view == "Transaction Lookup":
        render_lookup(raw_df, explainer)
    elif view == "Model Performance":
        render_performance()


if __name__ == "__main__":
    main()
