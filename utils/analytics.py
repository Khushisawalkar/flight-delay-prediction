"""
analytics.py — Aggregation and statistical analysis functions.

All functions accept a pandas DataFrame and return analysis-ready
dictionaries or DataFrames consumed by the visualization layer.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


# ── Dataset Overview ───────────────────────────────────────────────────────────
def dataset_overview(df: pd.DataFrame) -> Dict:
    """High-level summary statistics for the dataset."""
    total = len(df)
    delayed = int(df["is_delayed"].sum())
    on_time = total - delayed
    delay_rate = delayed / total if total > 0 else 0

    avg_delay = float(df.loc[df["is_delayed"] == 1, "delay_minutes"].mean()) \
        if "delay_minutes" in df.columns and delayed > 0 else 0

    return {
        "total_flights": total,
        "delayed_flights": delayed,
        "on_time_flights": on_time,
        "delay_rate": round(delay_rate, 4),
        "avg_delay_minutes": round(avg_delay, 1),
        "unique_airlines": df["airline"].nunique() if "airline" in df.columns else 0,
        "unique_airports": df["origin"].nunique() if "origin" in df.columns else 0,
        "unique_routes": (df["origin"] + "-" + df["dest"]).nunique()
            if all(c in df.columns for c in ["origin", "dest"]) else 0,
    }


# ── Airline Performance ────────────────────────────────────────────────────────
def airline_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-airline aggregation:
      - delay rate, avg delay minutes, flight count, on-time rate
    Sorted by delay rate descending.
    """
    grp = df.groupby("airline").agg(
        total_flights=("is_delayed", "count"),
        delayed_flights=("is_delayed", "sum"),
        avg_delay_min=("delay_minutes", "mean") if "delay_minutes" in df.columns else ("is_delayed", "sum"),
    ).reset_index()

    grp["delay_rate"] = grp["delayed_flights"] / grp["total_flights"]
    grp["on_time_rate"] = 1 - grp["delay_rate"]
    grp["delay_rate_pct"] = (grp["delay_rate"] * 100).round(1)
    grp["on_time_rate_pct"] = (grp["on_time_rate"] * 100).round(1)

    if "delay_minutes" in df.columns:
        delay_avg = df[df["is_delayed"] == 1].groupby("airline")["delay_minutes"].mean()
        grp["avg_delay_min"] = grp["airline"].map(delay_avg).fillna(0).round(1)

    return grp.sort_values("delay_rate", ascending=False).reset_index(drop=True)


# ── Airport Performance ────────────────────────────────────────────────────────
def airport_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-origin-airport aggregation with delay rates and flight volumes.
    """
    grp = df.groupby("origin").agg(
        departures=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
    ).reset_index()

    grp["delay_rate"] = grp["delayed"] / grp["departures"]
    grp["delay_rate_pct"] = (grp["delay_rate"] * 100).round(1)
    grp.rename(columns={"origin": "airport"}, inplace=True)

    return grp.sort_values("delay_rate", ascending=False).reset_index(drop=True)


# ── Hourly Delay Pattern ───────────────────────────────────────────────────────
def hourly_delay_pattern(df: pd.DataFrame) -> pd.DataFrame:
    """Delay rate by departure hour (0-23). Used for heatmap/line chart."""
    grp = df.groupby("dep_hour").agg(
        flights=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
    ).reset_index()
    grp["delay_rate"] = (grp["delayed"] / grp["flights"] * 100).round(1)
    grp["hour_label"] = grp["dep_hour"].apply(
        lambda h: f"{h:02d}:00" + (" AM" if h < 12 else " PM")
    )
    return grp.sort_values("dep_hour")


# ── Monthly Delay Trend ────────────────────────────────────────────────────────
def monthly_delay_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Delay rate + volume by month. Used for trend line."""
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    grp = df.groupby("dep_month").agg(
        flights=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
    ).reset_index()
    grp["delay_rate"] = (grp["delayed"] / grp["flights"] * 100).round(1)
    grp["month_name"] = grp["dep_month"].map(month_names)
    return grp.sort_values("dep_month")


# ── Route Analysis ────────────────────────────────────────────────────────────
def route_analysis(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """Top N routes by volume with delay metrics."""
    df = df.copy()
    df["route"] = df["origin"] + " → " + df["dest"]
    grp = df.groupby("route").agg(
        flights=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
    ).reset_index()
    grp["delay_rate"] = (grp["delayed"] / grp["flights"] * 100).round(1)
    return grp.sort_values("flights", ascending=False).head(top_n).reset_index(drop=True)


# ── Weather Impact Analysis ───────────────────────────────────────────────────
def weather_impact_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups by month to show weather-correlated delay patterns.
    Returns monthly weather risk vs actual delay rate.
    """
    weather_map = {1: 0.80, 2: 0.75, 3: 0.40, 4: 0.30, 5: 0.25, 6: 0.20,
                   7: 0.22, 8: 0.25, 9: 0.30, 10: 0.35, 11: 0.55, 12: 0.75}
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    grp = df.groupby("dep_month").agg(
        flights=("is_delayed", "count"),
        delayed=("is_delayed", "sum"),
    ).reset_index()
    grp["actual_delay_rate"] = (grp["delayed"] / grp["flights"] * 100).round(1)
    grp["weather_risk_index"] = (grp["dep_month"].map(weather_map) * 100).round(1)
    grp["month_name"] = grp["dep_month"].map(month_names)
    return grp.sort_values("dep_month")


# ── Delay Distribution ────────────────────────────────────────────────────────
def delay_distribution(df: pd.DataFrame) -> Dict:
    """
    Delay minutes distribution stats for delayed flights only.
    Returns histogram bins + percentile breakdown.
    """
    if "delay_minutes" not in df.columns:
        return {}

    delayed = df[df["is_delayed"] == 1]["delay_minutes"].dropna()
    if len(delayed) == 0:
        return {}

    buckets = {
        "15-30 min": int(((delayed >= 15) & (delayed < 30)).sum()),
        "30-60 min": int(((delayed >= 30) & (delayed < 60)).sum()),
        "1-2 hours": int(((delayed >= 60) & (delayed < 120)).sum()),
        "2-3 hours": int(((delayed >= 120) & (delayed < 180)).sum()),
        "3+ hours": int((delayed >= 180).sum()),
    }

    return {
        "buckets": buckets,
        "mean": round(float(delayed.mean()), 1),
        "median": round(float(delayed.median()), 1),
        "p75": round(float(delayed.quantile(0.75)), 1),
        "p90": round(float(delayed.quantile(0.90)), 1),
        "max": round(float(delayed.max()), 1),
        "values": delayed.clip(0, 300).tolist(),  # For histogram
    }


# ── Smart Insights Generator ──────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame, model_results: dict = None) -> List[str]:
    """
    Generates human-readable insights from dataset analytics.
    Returns list of markdown-formatted insight strings.
    """
    insights = []
    overview = dataset_overview(df)

    # Delay rate insight
    dr = overview["delay_rate"]
    if dr > 0.35:
        insights.append(f"🔴 **High delay rate**: {dr:.1%} of flights are delayed — significantly above the ~20% industry benchmark.")
    elif dr > 0.20:
        insights.append(f"🟡 **Moderate delays**: {dr:.1%} delay rate observed in this dataset.")
    else:
        insights.append(f"🟢 **Strong on-time performance**: Only {dr:.1%} of flights delayed.")

    # Worst airline
    ap = airline_performance(df)
    if len(ap) > 0:
        worst = ap.iloc[0]
        best = ap.iloc[-1]
        insights.append(
            f"✈️ **{worst['airline']}** has the highest delay rate at {worst['delay_rate_pct']}%, "
            f"while **{best['airline']}** leads with {best['on_time_rate_pct']}% on-time."
        )

    # Worst airport
    airp = airport_performance(df)
    if len(airp) > 0:
        worst_ap = airp.iloc[0]
        insights.append(
            f"🛫 **{worst_ap['airport']}** is the most delay-prone departure airport "
            f"({worst_ap['delay_rate_pct']}% delay rate across {worst_ap['departures']:,} departures)."
        )

    # Peak hour
    hp = hourly_delay_pattern(df)
    if len(hp) > 0:
        peak_hour = hp.loc[hp["delay_rate"].idxmax()]
        insights.append(
            f"🕐 **Worst departure hour**: {int(peak_hour['dep_hour']):02d}:00 with "
            f"{peak_hour['delay_rate']:.1f}% delay rate — likely due to cascade effects."
        )

    # Seasonal insight
    mp = monthly_delay_trend(df)
    if len(mp) > 0:
        worst_month = mp.loc[mp["delay_rate"].idxmax()]
        best_month = mp.loc[mp["delay_rate"].idxmin()]
        insights.append(
            f"📅 **{worst_month['month_name']}** is the worst month ({worst_month['delay_rate']:.1f}% delays), "
            f"while **{best_month['month_name']}** has the best on-time performance ({best_month['delay_rate']:.1f}%)."
        )

    # Model insight
    if model_results:
        best = max(model_results, key=lambda k: model_results[k].get("roc_auc", 0))
        auc = model_results[best].get("roc_auc", 0)
        insights.append(
            f"🤖 **{best}** is the top-performing model with AUC={auc:.3f} — "
            f"{'excellent' if auc > 0.85 else 'good'} discriminative power."
        )

    return insights