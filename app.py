"""
app.py — SkyMetrics: Aviation Delay Intelligence Platform
Aviation instrument-panel dark UI with amber/blue/radar-green palette.
Run: streamlit run app.py
"""

import os
import io
import json
import time
import pickle
import warnings
import tempfile
import numpy as np
import pandas as pd
import streamlit as st
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SkyMetrics · Flight Delay AI",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS: Aviation Instrument Panel ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;500;600;700&family=Barlow+Condensed:wght@400;600;700&display=swap');

/* ── Reset ── */
html, body, [class*="css"] {
    background: #07080e !important;
    color: #b8c4d8;
    font-family: 'Barlow', sans-serif;
}
* { box-sizing: border-box; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0c14 !important;
    border-right: 1px solid #141928;
    min-width: 260px !important;
}
section[data-testid="stSidebar"] * { color: #8a96b0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #5a6180 !important; font-size: 0.78rem; }

/* ── Header ── */
.sky-header {
    background: linear-gradient(135deg, #0a0c14 0%, #0d1020 60%, #0a0e1a 100%);
    border-bottom: 1px solid #1a2040;
    padding: 1.4rem 0 1.2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.sky-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(245,158,11,0.04) 0%, transparent 60%);
    pointer-events: none;
}
.sky-logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #e8eef8;
    text-align: center;
    line-height: 1;
}
.sky-logo .accent { color: #3b82f6; }
.sky-logo .accent2 { color: #f59e0b; }
.sky-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #3b5080;
    text-align: center;
    margin-top: 0.3rem;
    text-transform: uppercase;
}
.radar-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    margin-right: 8px;
    animation: radar-pulse 2s ease-in-out infinite;
}
@keyframes radar-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 6px #22c55e; }
    50% { opacity: 0.4; box-shadow: 0 0 14px #22c55e; }
}

/* ── Metric Cards ── */
.metrics-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.metric-card {
    flex: 1;
    min-width: 130px;
    background: #0d1020;
    border: 1px solid #1a2040;
    border-top: 2px solid;
    border-radius: 6px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 40px; height: 40px;
    background: radial-gradient(circle, rgba(255,255,255,0.03), transparent);
}
.metric-card.blue { border-top-color: #3b82f6; }
.metric-card.amber { border-top-color: #f59e0b; }
.metric-card.green { border-top-color: #22c55e; }
.metric-card.red { border-top-color: #ef4444; }
.metric-card.purple { border-top-color: #a855f7; }
.metric-val {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.7rem;
    font-weight: 400;
    color: #e8eef8;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #3d4a68;
    font-family: 'Barlow Condensed', sans-serif;
}
.metric-delta {
    font-size: 0.72rem;
    color: #4a5570;
    margin-top: 2px;
    font-family: 'Share Tech Mono', monospace;
}

/* ── Section Headers ── */
.section-hdr {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #3b5080;
    border-bottom: 1px solid #141928;
    padding-bottom: 8px;
    margin-bottom: 16px;
}

/* ── Prediction Card ── */
.pred-card {
    background: linear-gradient(135deg, #0d1020, #101525);
    border: 1px solid #1a2040;
    border-radius: 10px;
    padding: 24px;
    margin-bottom: 16px;
}
.pred-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    color: #3b5080;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.pred-result {
    font-family: 'Share Tech Mono', monospace;
    font-size: 3rem;
    font-weight: 400;
    line-height: 1;
    margin-bottom: 4px;
}
.pred-verdict {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 8px;
}

/* ── Insight Cards ── */
.insight-card {
    background: #0d1020;
    border: 1px solid #141928;
    border-left: 3px solid #3b82f6;
    border-radius: 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #8a96b0;
    animation: fadeIn 0.3s ease;
}
.insight-card b { color: #c8d3e8; }

/* ── Model Badge ── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #0d1020;
    border: 1px solid #1a2040;
    border-radius: 4px;
    padding: 4px 10px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    color: #3b82f6;
    margin: 2px;
}

/* ── Recommendation Card ── */
.rec-card {
    background: #0a0e1a;
    border: 1px solid #141928;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.85rem;
    line-height: 1.5;
    color: #8a96b0;
}

/* ── Training Status ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'Share Tech Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
.status-badge.trained { background: rgba(34,197,94,0.1); color: #22c55e; border: 1px solid rgba(34,197,94,0.2); }
.status-badge.pending { background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.2); }

/* ── Tabs ── */
.stTabs [data-testid="stTab"] {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #3d4a68 !important;
    padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: rgba(59,130,246,0.05) !important;
}
.stTabs [data-testid="stTabContent"] {
    background: transparent !important;
    border: 1px solid #141928 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 20px 16px !important;
}

/* ── Input fields ── */
.stTextInput input, .stNumberInput input {
    background: #0d1020 !important;
    border: 1px solid #1a2040 !important;
    border-radius: 6px !important;
    color: #c8d3e8 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.88rem !important;
}
.stSelectbox > div > div {
    background: #0d1020 !important;
    border: 1px solid #1a2040 !important;
    color: #c8d3e8 !important;
}
.stSlider [data-testid="stSlider"] { color: #3b82f6 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a2a4a, #1e3058) !important;
    color: #7aabf8 !important;
    border: 1px solid #2a4080 !important;
    border-radius: 6px !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
    padding: 8px 20px !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1e3058, #243870) !important;
    color: #a8c8f8 !important;
    border-color: #3b5898 !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a3a6a, #2050a0) !important;
    color: #88c0fc !important;
    border-color: #3b82f6 !important;
}

/* ── File Upload ── */
[data-testid="stFileUploader"] {
    background: #0d1020 !important;
    border: 1px dashed #1a2040 !important;
    border-radius: 8px !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    background: #0d1020 !important;
    border: 1px solid #141928 !important;
    border-radius: 6px !important;
}

/* ── Progress ── */
.stProgress > div > div { background-color: #3b82f6 !important; }

/* ── Dividers ── */
hr { border-color: #141928 !important; }

/* ── Alert / Info ── */
[data-testid="stAlert"] {
    background: #0d1020 !important;
    border: 1px solid #141928 !important;
    border-radius: 6px !important;
    color: #8a96b0 !important;
    font-size: 0.85rem !important;
}

/* ── Expander ── */
details {
    background: #0d1020 !important;
    border: 1px solid #141928 !important;
    border-radius: 6px !important;
}
summary { color: #5a6a90 !important; font-size: 0.82rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #07080e; }
::-webkit-scrollbar-thumb { background: #141928; border-radius: 2px; }

/* ── Animations ── */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #3b82f6 !important; }

/* ── Number/text inputs ── */
label[data-testid="stWidgetLabel"] {
    color: #5a6180 !important;
    font-size: 0.75rem !important;
    font-family: 'Barlow Condensed', sans-serif !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Main content border ── */
[data-testid="stAppViewContainer"] > .main > .block-container {
    max-width: 1400px;
    padding: 0 2rem 2rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────────
defaults = {
    "df": None,
    "model_metadata": None,
    "trained": False,
    "using_synthetic": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="sky-header">
    <div class="sky-logo">
        <span class="accent2">SKY</span>METRICS<span class="accent"> ✈</span>
    </div>
    <div class="sky-sub"><span class="radar-dot"></span>Aviation Delay Intelligence Platform · ML-Powered</div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-hdr">⚙ Data Source</div>', unsafe_allow_html=True)

    data_source = st.radio(
        "Select Data Source",
        ["🛰 Generate Synthetic Data", "📂 Upload CSV Dataset"],
        label_visibility="collapsed",
    )

    if data_source == "📂 Upload CSV Dataset":
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if uploaded:
            from utils.preprocessing import normalize_uploaded_csv
            raw_df = pd.read_csv(uploaded)
            st.session_state.df = normalize_uploaded_csv(raw_df)
            st.success(f"✅ {len(st.session_state.df):,} rows loaded")

    n_rows = 15000
    if data_source == "🛰 Generate Synthetic Data":
        n_rows = st.select_slider(
            "Dataset Size",
            options=[5000, 10000, 15000, 25000],
            value=15000,
        )

    st.markdown('<div class="section-hdr" style="margin-top:16px;">🤖 Model Training</div>', unsafe_allow_html=True)

    if st.button("⚡ Train All Models", use_container_width=True, type="primary"):
        from train_model import train_all_models
        from utils.preprocessing import generate_synthetic_dataset

        with st.spinner("Training Random Forest, XGBoost, Logistic Regression..."):
            progress = st.progress(0, text="Generating dataset...")

            if st.session_state.df is None or data_source == "🛰 Generate Synthetic Data":
                df = generate_synthetic_dataset(n_rows=n_rows)
                st.session_state.df = df
                st.session_state.using_synthetic = True
            else:
                df = st.session_state.df

            progress.progress(0.2, text="Engineering features...")
            time.sleep(0.3)
            progress.progress(0.3, text="Training models...")
            metadata = train_all_models(df=df, verbose=False)
            progress.progress(1.0, text="Done!")
            time.sleep(0.5)
            progress.empty()

        st.session_state.model_metadata = metadata
        st.session_state.trained = True
        st.success("✅ Models trained & saved")

    # Training status
    if st.session_state.trained:
        st.markdown('<span class="status-badge trained">● Models Ready</span>', unsafe_allow_html=True)
    else:
        # Try loading existing
        from pathlib import Path
        if (Path("models") / "metadata.json").exists():
            import json
            with open("models/metadata.json") as f:
                st.session_state.model_metadata = json.load(f)
            st.session_state.trained = True
            st.markdown('<span class="status-badge trained">● Models Loaded</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge pending">○ Not Trained</span>', unsafe_allow_html=True)

    if st.session_state.model_metadata:
        best = st.session_state.model_metadata.get("best_model", "—")
        results = st.session_state.model_metadata.get("results", {})
        auc = results.get(best, {}).get("roc_auc", 0)
        st.markdown(
            f'<div class="model-badge">🏆 {best}</div>'
            f'<div class="model-badge">AUC {auc:.3f}</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown('<div class="section-hdr">📊 Chart Settings</div>', unsafe_allow_html=True)
    top_n_routes = st.slider("Top Routes to Show", 5, 20, 10)
    show_raw_data = st.checkbox("Show Raw Data Table", value=False)


# ── Ensure we have data ───────────────────────────────────────────────────────
if st.session_state.df is None:
    from utils.preprocessing import generate_synthetic_dataset
    with st.spinner("Loading sample dataset..."):
        st.session_state.df = generate_synthetic_dataset(n_rows=10000)
        st.session_state.using_synthetic = True

df = st.session_state.df

# ── Precompute analytics ──────────────────────────────────────────────────────
from utils.analytics import (
    dataset_overview, airline_performance, airport_performance,
    hourly_delay_pattern, monthly_delay_trend, route_analysis,
    weather_impact_analysis, delay_distribution, generate_insights
)

overview = dataset_overview(df)
airline_df = airline_performance(df)
airport_df = airport_performance(df)
hourly_df = hourly_delay_pattern(df)
monthly_df = monthly_delay_trend(df)
route_df = route_analysis(df, top_n=top_n_routes)
weather_df = weather_impact_analysis(df)
delay_dist = delay_distribution(df)


# ── KPI Strip ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
kpis = [
    (c1, overview["total_flights"], "Total Flights", f"{overview['unique_routes']} routes", "blue"),
    (c2, f"{overview['delay_rate']:.1%}", "Delay Rate", f"{overview['delayed_flights']:,} delayed", "red"),
    (c3, f"{overview['avg_delay_minutes']} min", "Avg Delay", "Delayed flights only", "amber"),
    (c4, overview["unique_airlines"], "Airlines", "Tracked carriers", "green"),
    (c5, overview["unique_airports"], "Airports", "Origin airports", "purple"),
]
for col, val, label, delta, color in kpis:
    with col:
        st.markdown(f"""
        <div class="metric-card {color}">
            <div class="metric-val">{val:,}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-delta">{delta}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Tabs ─────────────────────────────────────────────────────────────────
from utils.visualizations import (
    chart_airline_delay_rates, chart_hourly_pattern, chart_monthly_weather_impact,
    chart_confusion_matrix, chart_roc_curves, chart_feature_importance,
    chart_model_comparison, chart_delay_distribution, chart_airport_heatmap,
    chart_delay_gauge, chart_route_scatter, chart_contribution_waterfall
)

tabs = st.tabs([
    "📡 Live Prediction",
    "📊 Airline Analytics",
    "🛫 Airport Analysis",
    "🌦 Weather Impact",
    "🗺 Route Explorer",
    "🤖 ML Performance",
    "💡 AI Insights",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE PREDICTION
# ════════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-hdr">✈ Flight Delay Prediction Engine</div>', unsafe_allow_html=True)

    if not st.session_state.trained:
        st.warning("⚠️ Train models first using the sidebar. Using rule-based prediction fallback.")

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        from utils.preprocessing import AIRLINES, AIRPORTS

        st.markdown("**Flight Parameters**")

        airline = st.selectbox("Airline", AIRLINES)
        c1_, c2_ = st.columns(2)
        with c1_:
            origin = st.selectbox("Origin Airport", AIRPORTS)
        with c2_:
            dest = st.selectbox("Destination", [a for a in AIRPORTS if a != origin])

        c3_, c4_ = st.columns(2)
        with c3_:
            dep_month = st.selectbox("Month", range(1, 13),
                                     format_func=lambda m: ["","Jan","Feb","Mar","Apr","May","Jun",
                                                             "Jul","Aug","Sep","Oct","Nov","Dec"][m])
        with c4_:
            dep_day = st.slider("Day of Month", 1, 28, 15)

        dep_hour = st.slider("Departure Hour (24h)", 5, 23, 10,
                             format="%d:00")

        c5_, c6_ = st.columns(2)
        with c5_:
            distance = st.number_input("Distance (miles)", 100, 3000, 800, 50)
        with c6_:
            elapsed = st.number_input("Sched. Duration (min)", 45, 600, 150, 10)

        predict_btn = st.button("🔍 Predict Delay Risk", type="primary", use_container_width=True)

    with col_result:
        if predict_btn:
            if not st.session_state.trained:
                # Fallback: rule-based estimate
                from utils.prediction import compute_feature_contributions, generate_recommendations
                from utils.preprocessing import AIRLINE_RELIABILITY, DELAY_PRONE_AIRPORTS
                weather_map = {1:0.8,2:0.75,3:0.4,4:0.3,5:0.25,6:0.2,
                               7:0.22,8:0.25,9:0.3,10:0.35,11:0.55,12:0.75}
                prob = (0.20
                       + (1 - AIRLINE_RELIABILITY.get(airline, 0.72)) * 0.45
                       + DELAY_PRONE_AIRPORTS.get(origin, 0.15) * 0.40
                       + weather_map.get(dep_month, 0.3) * 0.30
                       + (0.10 if (7<=dep_hour<=9 or 17<=dep_hour<=20) else 0)
                       + (0.07 if dep_month in (12,1,11) else 0))
                prob = min(max(prob, 0.05), 0.95)
                risk_color = "#22c55e" if prob<0.3 else "#f59e0b" if prob<0.55 else "#ef4444"
                risk_level = "Low" if prob<0.3 else "Medium" if prob<0.55 else "High" if prob<0.75 else "Critical"
                result = {
                    "delay_probability": prob,
                    "delay_percent": round(prob*100, 1),
                    "is_delayed": prob >= 0.5,
                    "risk_level": risk_level,
                    "risk_color": risk_color,
                    "feature_contributions": [],
                    "recommendations": generate_recommendations({
                        "is_peak_hour": int(7<=dep_hour<=9 or 17<=dep_hour<=20),
                        "weather_risk": weather_map.get(dep_month, 0.3),
                        "airport_congestion": DELAY_PRONE_AIRPORTS.get(origin, 0.15),
                        "carrier_delay_hist": 1-AIRLINE_RELIABILITY.get(airline, 0.72),
                        "is_holiday_season": int(dep_month in (12,1)),
                    }, prob, airline, origin, dep_month, dep_hour),
                    "model_used": "Rule-Based",
                }
            else:
                with st.spinner("Computing delay probability..."):
                    from utils.prediction import predict_delay
                    result = predict_delay(
                        airline, origin, dest, dep_month, dep_day,
                        dep_hour, distance, elapsed
                    )

            # Result display
            verdict = "LIKELY DELAYED" if result["is_delayed"] else "ON TIME"
            rc = result["risk_color"]

            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-title">Prediction Result · {result.get('model_used','AI Model')}</div>
                <div class="pred-result" style="color:{rc}">{result['delay_percent']}%</div>
                <div class="pred-verdict" style="color:{rc}">▶ {verdict}</div>
                <div style="margin-top:12px; font-size:0.75rem; color:#3d4a68; font-family:'Share Tech Mono',monospace;">
                    RISK LEVEL: <span style="color:{rc}">{result['risk_level'].upper()}</span>
                    &nbsp;|&nbsp; ROUTE: {origin}→{dest}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge
            st.plotly_chart(
                chart_delay_gauge(result["delay_probability"], result["risk_level"], result["risk_color"]),
                use_container_width=True, config={"displayModeBar": False}
            )

            # Recommendations
            if result["recommendations"]:
                st.markdown('<div class="section-hdr">💡 Recommendations</div>', unsafe_allow_html=True)
                for rec in result["recommendations"]:
                    st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)

    # Feature contributions waterfall
    if predict_btn and result.get("feature_contributions"):
        st.markdown('<div class="section-hdr" style="margin-top:16px;">🔬 Prediction Breakdown</div>',
                    unsafe_allow_html=True)
        fig_wf = chart_contribution_waterfall(result["feature_contributions"])
        st.plotly_chart(fig_wf, use_container_width=True, config={"displayModeBar": False})

    elif not predict_btn:
        st.markdown("""
        <div style="text-align:center; padding:3rem; color:#2a3250; font-family:'Share Tech Mono',monospace; font-size:0.82rem;">
            <div style="font-size:2rem;margin-bottom:12px;">📡</div>
            AWAITING FLIGHT PARAMETERS<br>
            <span style="font-size:0.68rem;color:#1a2038">Configure flight above and click PREDICT</span>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — AIRLINE ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-hdr">✈ Airline Performance Analysis</div>', unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.plotly_chart(chart_airline_delay_rates(airline_df), use_container_width=True,
                        config={"displayModeBar": False})
    with col_b:
        st.markdown('<div class="section-hdr">Rankings</div>', unsafe_allow_html=True)
        display_df = airline_df[["airline", "total_flights", "delay_rate_pct", "on_time_rate_pct"]].copy()
        display_df.columns = ["Airline", "Flights", "Delay%", "OnTime%"]
        st.dataframe(
            display_df.style.background_gradient(subset=["Delay%"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )

    st.plotly_chart(chart_hourly_pattern(hourly_df), use_container_width=True,
                    config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — AIRPORT ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-hdr">🛫 Airport Departure Performance</div>', unsafe_allow_html=True)

    col_ap1, col_ap2 = st.columns([2, 3])
    with col_ap1:
        top_airport_df = airport_df.head(15)
        st.dataframe(
            top_airport_df[["airport", "departures", "delay_rate_pct"]].rename(
                columns={"airport": "Airport", "departures": "Departures", "delay_rate_pct": "Delay%"}
            ).style.background_gradient(subset=["Delay%"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True,
        )
    with col_ap2:
        st.plotly_chart(chart_airport_heatmap(airport_df.head(15)),
                        use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — WEATHER IMPACT
# ════════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-hdr">🌦 Weather & Seasonal Delay Analysis</div>', unsafe_allow_html=True)

    st.plotly_chart(chart_monthly_weather_impact(weather_df),
                    use_container_width=True, config={"displayModeBar": False})

    if delay_dist.get("values"):
        col_d1, col_d2 = st.columns([3, 1])
        with col_d1:
            st.plotly_chart(chart_delay_distribution(delay_dist["values"]),
                            use_container_width=True, config={"displayModeBar": False})
        with col_d2:
            st.markdown('<div class="section-hdr">Delay Stats</div>', unsafe_allow_html=True)
            stats = [
                ("Mean", f"{delay_dist['mean']} min"),
                ("Median", f"{delay_dist['median']} min"),
                ("P75", f"{delay_dist['p75']} min"),
                ("P90", f"{delay_dist['p90']} min"),
                ("Max", f"{delay_dist['max']} min"),
            ]
            for label, val in stats:
                st.markdown(f"""
                <div class="metric-card amber" style="margin-bottom:8px;padding:10px 14px;">
                    <div class="metric-val" style="font-size:1.2rem">{val}</div>
                    <div class="metric-label">{label} Delay</div>
                </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROUTE EXPLORER
# ════════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-hdr">🗺 Route Performance Explorer</div>', unsafe_allow_html=True)

    st.plotly_chart(chart_route_scatter(route_df), use_container_width=True,
                    config={"displayModeBar": False})

    st.markdown('<div class="section-hdr">Top Routes by Volume</div>', unsafe_allow_html=True)
    st.dataframe(
        route_df[["route", "flights", "delayed", "delay_rate"]].rename(
            columns={"route": "Route", "flights": "Flights", "delayed": "Delayed", "delay_rate": "Delay%"}
        ).style.background_gradient(subset=["Delay%"], cmap="RdYlGn_r"),
        use_container_width=True, hide_index=True,
    )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — ML PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-hdr">🤖 Machine Learning Model Evaluation</div>', unsafe_allow_html=True)

    if not st.session_state.model_metadata:
        st.info("Train models using the sidebar to view ML performance metrics.")
    else:
        results = st.session_state.model_metadata.get("results", {})
        feature_names = st.session_state.model_metadata.get("feature_names", [])
        best_model = st.session_state.model_metadata.get("best_model", "")

        # Metrics comparison
        st.plotly_chart(chart_model_comparison(results), use_container_width=True,
                        config={"displayModeBar": False})

        col_m1, col_m2 = st.columns(2)

        with col_m1:
            # ROC curves
            st.plotly_chart(chart_roc_curves(results), use_container_width=True,
                            config={"displayModeBar": False})

        with col_m2:
            # Model selector for confusion matrix
            selected = st.selectbox("Select Model", list(results.keys()))
            cm = results[selected].get("confusion_matrix", [[0,0],[0,0]])
            st.plotly_chart(chart_confusion_matrix(cm, selected),
                            use_container_width=True, config={"displayModeBar": False})

        # Feature importance
        st.markdown('<div class="section-hdr">🔍 Feature Importances</div>', unsafe_allow_html=True)
        fi_col1, fi_col2 = st.columns(2)
        model_list = list(results.items())

        for i, (name, res) in enumerate(model_list[:2]):
            col = fi_col1 if i == 0 else fi_col2
            with col:
                importances = res.get("feature_importances", {})
                if importances:
                    st.plotly_chart(chart_feature_importance(importances, name),
                                    use_container_width=True, config={"displayModeBar": False})

        # Metrics table
        st.markdown('<div class="section-hdr">Metrics Summary</div>', unsafe_allow_html=True)
        metrics_rows = []
        for name, res in results.items():
            metrics_rows.append({
                "Model": name,
                "Accuracy": f"{res.get('accuracy', 0):.4f}",
                "F1 Score": f"{res.get('f1_score', 0):.4f}",
                "ROC AUC": f"{res.get('roc_auc', 0):.4f}",
                "Precision": f"{res.get('precision', 0):.4f}",
                "Recall": f"{res.get('recall', 0):.4f}",
                "Train Time": f"{res.get('train_time_sec', 0):.1f}s",
                "Best": "⭐" if name == best_model else "",
            })
        st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 7 — AI INSIGHTS
# ════════════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-hdr">💡 AI-Generated Insights & Recommendations</div>',
                unsafe_allow_html=True)

    model_results = st.session_state.model_metadata.get("results") \
        if st.session_state.model_metadata else None
    insights = generate_insights(df, model_results)

    for insight in insights:
        st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">📋 Dataset Summary</div>', unsafe_allow_html=True)

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        # Best & worst airline
        st.markdown("**Airline Performance Extremes**")
        best_al = airline_df.iloc[-1]
        worst_al = airline_df.iloc[0]
        st.markdown(f"""
        <div class="rec-card">
            🏆 <b>Best:</b> {best_al['airline']} — {best_al['on_time_rate_pct']}% on-time<br>
            ⚠️ <b>Worst:</b> {worst_al['airline']} — {worst_al['delay_rate_pct']}% delay rate
        </div>""", unsafe_allow_html=True)

        # Monthly trend
        worst_m = monthly_df.loc[monthly_df["delay_rate"].idxmax()]
        best_m = monthly_df.loc[monthly_df["delay_rate"].idxmin()]
        st.markdown(f"""
        <div class="rec-card">
            📅 Peak month: <b>{worst_m['month_name']}</b> ({worst_m['delay_rate']:.1f}% delays)<br>
            ✅ Best month: <b>{best_m['month_name']}</b> ({best_m['delay_rate']:.1f}% delays)
        </div>""", unsafe_allow_html=True)

    with col_i2:
        # Delay buckets
        if delay_dist.get("buckets"):
            st.markdown("**Delay Duration Breakdown**")
            for bucket, count in delay_dist["buckets"].items():
                pct = count / overview["delayed_flights"] * 100 if overview["delayed_flights"] > 0 else 0
                st.markdown(f"""
                <div class="rec-card" style="padding:8px 12px;">
                    <b>{bucket}</b> — {count:,} flights ({pct:.1f}%)
                </div>""", unsafe_allow_html=True)

    # Raw data
    if show_raw_data:
        st.markdown('<div class="section-hdr">🗃 Raw Dataset Sample</div>', unsafe_allow_html=True)
        st.dataframe(df.head(500), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem;
            font-family:'Share Tech Mono',monospace; font-size:0.65rem;
            color:#1a2038; letter-spacing:0.1em;">
    SKYMETRICS AVIATION INTELLIGENCE PLATFORM · ML-POWERED DELAY PREDICTION
    · RANDOM FOREST · XGBOOST · LOGISTIC REGRESSION · FAISS-FREE RAG
</div>
""", unsafe_allow_html=True)