"""
visualizations.py — All Plotly chart builders.

Aviation-themed dark palette:
  Background: #0a0c12   Surface: #111520   Grid: #1e2235
  Accent 1:   #3b82f6  (electric blue)
  Accent 2:   #f59e0b  (amber/cockpit gold)
  Accent 3:   #22c55e  (radar green)
  Accent 4:   #ef4444  (warning red)
  Text:       #c8d3e8
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# ── Theme Constants ────────────────────────────────────────────────────────────
BG = "#0a0c12"
SURFACE = "#111520"
SURFACE2 = "#161b2c"
GRID = "#1e2235"
TEXT = "#c8d3e8"
TEXT_DIM = "#5a6180"
BLUE = "#3b82f6"
AMBER = "#f59e0b"
GREEN = "#22c55e"
RED = "#ef4444"
ORANGE = "#f97316"
PURPLE = "#a855f7"

LAYOUT_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=SURFACE,
    font=dict(family="'Share Tech Mono', monospace", color=TEXT, size=11),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=TEXT_DIM)),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, tickfont=dict(color=TEXT_DIM)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_DIM)),
)

COLOR_SCALE = [[0, "#0a2a4a"], [0.33, BLUE], [0.66, AMBER], [1.0, RED]]


def _apply_base(fig, title: str = "") -> go.Figure:
    fig.update_layout(**LAYOUT_BASE, title=dict(
        text=title, font=dict(color=TEXT, size=13), x=0.02
    ))
    return fig


# ── 1. Airline Performance Bar ─────────────────────────────────────────────────
def chart_airline_delay_rates(airline_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    colors = [RED if r > 40 else AMBER if r > 25 else GREEN
              for r in airline_df["delay_rate_pct"]]

    fig.add_trace(go.Bar(
        x=airline_df["airline"],
        y=airline_df["delay_rate_pct"],
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in airline_df["delay_rate_pct"]],
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="<b>%{x}</b><br>Delay Rate: %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(y=airline_df["delay_rate_pct"].mean(),
                  line_dash="dash", line_color=BLUE, line_width=1,
                  annotation_text=f"Avg {airline_df['delay_rate_pct'].mean():.1f}%",
                  annotation_font_color=BLUE)

    _apply_base(fig, "✈ Airline Delay Rates (%)")
    fig.update_yaxis(title_text="Delay Rate (%)", title_font_color=TEXT_DIM)
    return fig


# ── 2. Hourly Heatmap / Line ───────────────────────────────────────────────────
def chart_hourly_pattern(hourly_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    # Fill area
    fig.add_trace(go.Scatter(
        x=hourly_df["dep_hour"],
        y=hourly_df["delay_rate"],
        fill="tozeroy",
        fillcolor=f"rgba(59,130,246,0.12)",
        line=dict(color=BLUE, width=2),
        mode="lines+markers",
        marker=dict(color=hourly_df["delay_rate"],
                    colorscale=[[0, GREEN], [0.5, AMBER], [1, RED]],
                    size=8, line=dict(width=0)),
        hovertemplate="<b>%{x}:00</b><br>Delay Rate: %{y:.1f}%<extra></extra>",
        name="Delay Rate",
    ))

    # Mark peak hours
    for h in [7, 8, 9, 17, 18, 19, 20]:
        fig.add_vline(x=h, line_color=f"rgba(245,158,11,0.15)", line_width=1)

    _apply_base(fig, "🕐 Delay Rate by Departure Hour")
    fig.update_xaxis(title_text="Hour of Day", tickmode="linear", dtick=2)
    fig.update_yaxis(title_text="Delay Rate (%)")
    return fig


# ── 3. Monthly Trend with Weather Overlay ─────────────────────────────────────
def chart_monthly_weather_impact(monthly_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=monthly_df["month_name"],
        y=monthly_df["actual_delay_rate"],
        name="Actual Delay Rate %",
        marker_color=BLUE,
        opacity=0.75,
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ), secondary_y=False)

    if "weather_risk_index" in monthly_df.columns:
        fig.add_trace(go.Scatter(
            x=monthly_df["month_name"],
            y=monthly_df["weather_risk_index"],
            name="Weather Risk Index",
            line=dict(color=AMBER, width=2, dash="dot"),
            mode="lines+markers",
            marker=dict(size=6),
            hovertemplate="%{x}: Risk=%{y:.1f}<extra></extra>",
        ), secondary_y=True)

    _apply_base(fig, "🌦 Monthly Delays vs Weather Risk")
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=SURFACE,
                      font=dict(family="'Share Tech Mono', monospace", color=TEXT, size=11),
                      margin=dict(l=40, r=40, t=50, b=40),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_DIM)))
    fig.update_yaxes(title_text="Delay Rate (%)", gridcolor=GRID, secondary_y=False)
    fig.update_yaxes(title_text="Weather Risk Index", gridcolor=GRID, secondary_y=True)
    return fig


# ── 4. Confusion Matrix ────────────────────────────────────────────────────────
def chart_confusion_matrix(cm: list, model_name: str) -> go.Figure:
    z = cm
    labels = ["On-Time", "Delayed"]

    annots = [[f"<b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale=[[0, SURFACE2], [0.5, "#1e3a5f"], [1.0, BLUE]],
        showscale=False,
        text=annots,
        texttemplate="%{text}",
        textfont=dict(size=20, color="white"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))

    _apply_base(fig, f"Confusion Matrix — {model_name}")
    fig.update_xaxis(title_text="Predicted", side="bottom")
    fig.update_yaxis(title_text="Actual", autorange="reversed")
    return fig


# ── 5. ROC Curves (all models) ─────────────────────────────────────────────────
def chart_roc_curves(model_results: dict) -> go.Figure:
    fig = go.Figure()

    palette = [BLUE, AMBER, GREEN, ORANGE, PURPLE]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(model_results)}

    for name, res in model_results.items():
        if "roc_fpr" not in res:
            continue
        auc = res.get("roc_auc", 0)
        fig.add_trace(go.Scatter(
            x=res["roc_fpr"],
            y=res["roc_tpr"],
            name=f"{name} (AUC={auc:.3f})",
            line=dict(color=colors[name], width=2),
            hovertemplate=f"{name}<br>FPR: %{{x:.3f}}<br>TPR: %{{y:.3f}}<extra></extra>",
        ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(color=GRID, width=1, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))

    _apply_base(fig, "📈 ROC Curves — Model Comparison")
    fig.update_xaxis(title_text="False Positive Rate", range=[0, 1])
    fig.update_yaxis(title_text="True Positive Rate", range=[0, 1])
    return fig


# ── 6. Feature Importance ─────────────────────────────────────────────────────
def chart_feature_importance(importances: dict, model_name: str, top_n: int = 12) -> go.Figure:
    if not importances:
        return go.Figure()

    sorted_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, vals = zip(*sorted_feats)

    # Friendly labels
    label_map = {
        "carrier_delay_hist": "Carrier Delay History",
        "airport_congestion": "Airport Congestion",
        "weather_risk": "Weather Risk",
        "dep_hour": "Departure Hour",
        "dep_month": "Month",
        "distance": "Distance",
        "is_peak_hour": "Peak Hour Flag",
        "is_holiday_season": "Holiday Season",
        "route_delay_rate": "Route Delay Rate",
        "dep_hour_sin": "Hour (sin)",
        "dep_hour_cos": "Hour (cos)",
        "month_sin": "Month (sin)",
        "month_cos": "Month (cos)",
        "airline_enc": "Airline",
        "origin_enc": "Origin Airport",
        "dest_enc": "Dest Airport",
        "is_weekend": "Weekend",
        "scheduled_elapsed": "Flight Duration",
        "dep_dayofweek": "Day of Week",
    }
    friendly_names = [label_map.get(n, n.replace("_", " ").title()) for n in names]

    # Color gradient by importance
    max_val = max(vals)
    colors = [f"rgba(59,130,246,{0.4 + 0.6 * v / max_val:.2f})" for v in vals]

    fig = go.Figure(go.Bar(
        x=list(vals)[::-1],
        y=friendly_names[::-1],
        orientation="h",
        marker=dict(color=list(colors)[::-1], line=dict(width=0)),
        text=[f"{v:.4f}" for v in list(vals)[::-1]],
        textposition="outside",
        textfont=dict(color=TEXT_DIM, size=9),
        hovertemplate="%{y}: %{x:.5f}<extra></extra>",
    ))

    _apply_base(fig, f"🔍 Feature Importance — {model_name}")
    fig.update_xaxis(title_text="Importance Score")
    fig.update_layout(height=420)
    return fig


# ── 7. Model Comparison Radar / Bar ───────────────────────────────────────────
def chart_model_comparison(model_results: dict) -> go.Figure:
    metrics = ["accuracy", "f1_score", "roc_auc", "precision", "recall"]
    labels = ["Accuracy", "F1 Score", "ROC AUC", "Precision", "Recall"]
    palette = [BLUE, AMBER, GREEN, ORANGE, PURPLE]

    fig = go.Figure()
    for i, (name, res) in enumerate(model_results.items()):
        vals = [res.get(m, 0) for m in metrics]
        color = palette[i % len(palette)]
        fig.add_trace(go.Bar(
            name=name,
            x=labels,
            y=vals,
            marker_color=color,
            opacity=0.85,
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(color=TEXT, size=9),
            hovertemplate=f"{name}<br>%{{x}}: %{{y:.4f}}<extra></extra>",
        ))

    _apply_base(fig, "🤖 ML Model Performance Comparison")
    fig.update_layout(barmode="group")
    fig.update_yaxis(title_text="Score", range=[0, 1.1])
    return fig


# ── 8. Delay Distribution Histogram ───────────────────────────────────────────
def chart_delay_distribution(values: list) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=values,
        nbinsx=40,
        marker=dict(
            color=BLUE, opacity=0.75,
            line=dict(color=SURFACE2, width=0.5)
        ),
        hovertemplate="Delay: %{x} min<br>Flights: %{y}<extra></extra>",
    ))
    _apply_base(fig, "📊 Delay Duration Distribution (Delayed Flights Only)")
    fig.update_xaxis(title_text="Delay Minutes")
    fig.update_yaxis(title_text="Number of Flights")
    return fig


# ── 9. Airport Delay Heatmap ──────────────────────────────────────────────────
def chart_airport_heatmap(airport_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=airport_df["delay_rate_pct"],
        y=airport_df["airport"],
        orientation="h",
        marker=dict(
            color=airport_df["delay_rate_pct"],
            colorscale=[[0, GREEN], [0.5, AMBER], [1.0, RED]],
            line=dict(width=0),
        ),
        text=[f"{v:.1f}%" for v in airport_df["delay_rate_pct"]],
        textposition="outside",
        textfont=dict(color=TEXT, size=9),
        hovertemplate="<b>%{y}</b><br>Delay Rate: %{x:.1f}%<extra></extra>",
    ))
    _apply_base(fig, "🛫 Airport Delay Rate Ranking")
    fig.update_xaxis(title_text="Delay Rate (%)")
    fig.update_layout(height=max(350, len(airport_df) * 28))
    return fig


# ── 10. Delay Probability Gauge ───────────────────────────────────────────────
def chart_delay_gauge(probability: float, risk_level: str, risk_color: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number=dict(suffix="%", font=dict(size=42, color=risk_color,
                    family="'Share Tech Mono', monospace")),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor=GRID,
                      tickfont=dict(color=TEXT_DIM)),
            bar=dict(color=risk_color, thickness=0.25),
            bgcolor=SURFACE2,
            borderwidth=0,
            steps=[
                dict(range=[0, 30], color="#0a2a1a"),
                dict(range=[30, 55], color="#2a2a0a"),
                dict(range=[55, 75], color="#2a1a0a"),
                dict(range=[75, 100], color="#2a0a0a"),
            ],
            threshold=dict(
                line=dict(color=risk_color, width=3),
                thickness=0.8,
                value=probability * 100,
            ),
        ),
        title=dict(text=f"DELAY PROBABILITY<br><span style='font-size:14px;color:{risk_color}'>"
                        f"▶ {risk_level.upper()} RISK</span>",
                   font=dict(size=13, color=TEXT)),
    ))

    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(family="'Share Tech Mono', monospace", color=TEXT),
        margin=dict(l=20, r=20, t=60, b=20),
        height=300,
    )
    return fig


# ── 11. Route Analysis Scatter ────────────────────────────────────────────────
def chart_route_scatter(route_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=route_df["flights"],
        y=route_df["delay_rate"],
        mode="markers+text",
        marker=dict(
            size=np.clip(route_df["flights"] / route_df["flights"].max() * 30 + 6, 6, 30),
            color=route_df["delay_rate"],
            colorscale=[[0, GREEN], [0.5, AMBER], [1.0, RED]],
            line=dict(color=SURFACE, width=1),
            showscale=True,
            colorbar=dict(title="Delay%", tickfont=dict(color=TEXT_DIM)),
        ),
        text=route_df["route"],
        textposition="top center",
        textfont=dict(size=8, color=TEXT_DIM),
        hovertemplate="<b>%{text}</b><br>Flights: %{x:,}<br>Delay Rate: %{y:.1f}%<extra></extra>",
    ))
    _apply_base(fig, "🗺 Route Volume vs Delay Rate")
    fig.update_xaxis(title_text="Flight Volume")
    fig.update_yaxis(title_text="Delay Rate (%)")
    return fig


# ── 12. Feature Contribution Waterfall (Prediction) ──────────────────────────
def chart_contribution_waterfall(contributions: list, baseline: float = 0.20) -> go.Figure:
    if not contributions:
        return go.Figure()

    names = ["Baseline"] + [c["factor"] for c in contributions]
    values = [baseline] + [c["impact"] for c in contributions]

    colors = [BLUE] + [GREEN if v < 0 else RED for v in values[1:]]

    fig = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        text=[f"{'+' if v > 0 else ''}{v:.3f}" for v in values],
        textposition="outside",
        textfont=dict(color=TEXT, size=10),
        hovertemplate="%{x}<br>Impact: %{y:.4f}<extra></extra>",
    ))

    _apply_base(fig, "🔬 Prediction Factor Breakdown")
    fig.update_yaxis(title_text="Impact on Delay Probability")
    return fig