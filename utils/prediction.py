"""
prediction.py — Live inference engine.

Loads trained model + scaler, builds feature vector from user inputs,
returns delay probability + SHAP-style feature contribution breakdown.
"""

import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from utils.preprocessing import engineer_features, FEATURE_COLS, AIRLINE_RELIABILITY, DELAY_PRONE_AIRPORTS

MODELS_DIR = Path("models")


# ── Model Loader (cached) ──────────────────────────────────────────────────────
_cache = {}

def load_artifacts():
    """Load model + scaler + metadata once and cache in memory."""
    if "model" in _cache:
        return _cache["model"], _cache["scaler"], _cache["metadata"]

    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError("Models not trained yet. Run train_model.py first.")

    with open(meta_path) as f:
        metadata = json.load(f)

    best_name = metadata["best_model"]
    safe_name = best_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_path = MODELS_DIR / f"{safe_name}.pkl"

    if not model_path.exists():
        model_path = MODELS_DIR / "best_model.pkl"

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    _cache["model"] = model
    _cache["scaler"] = scaler
    _cache["metadata"] = metadata
    return model, scaler, metadata


def load_all_models():
    """Load all trained models for comparison UI."""
    meta_path = MODELS_DIR / "metadata.json"
    if not meta_path.exists():
        return {}

    with open(meta_path) as f:
        metadata = json.load(f)

    models = {}
    for name in metadata.get("results", {}).keys():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        path = MODELS_DIR / f"{safe_name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


# ── Feature Vector Builder ──────────────────────────────────────────────────────
def build_feature_vector(
    airline: str,
    origin: str,
    dest: str,
    dep_month: int,
    dep_day: int,
    dep_hour: int,
    distance: int,
    scheduled_elapsed: int,
    feature_names: list,
    scaler,
) -> tuple:
    """
    Constructs feature vector for a single flight.
    Returns (scaled_vector, raw_feature_dict).
    """
    from sklearn.preprocessing import LabelEncoder

    # Build a single-row dataframe matching training schema
    row = pd.DataFrame([{
        "airline": airline,
        "origin": origin,
        "dest": dest,
        "dep_month": dep_month,
        "dep_day": dep_day,
        "dep_hour": dep_hour,
        "distance": distance,
        "scheduled_elapsed": scheduled_elapsed,
        "is_delayed": 0,  # Placeholder for feature engineering
    }])

    # We need reference data to compute route_delay_rate, so use global averages
    row["route"] = row["origin"] + "_" + row["dest"]

    # Manual feature engineering (mirrors preprocessing.py)
    from sklearn.preprocessing import LabelEncoder
    all_airlines = list(AIRLINE_RELIABILITY.keys())
    all_airports = list(set(DELAY_PRONE_AIRPORTS.keys()) | {"ATL", "LAX", "DFW", "DEN", "JFK",
                        "SFO", "SEA", "LAS", "MCO", "MIA", "PHX", "IAH", "CLT", "BOS"})

    le_airline = LabelEncoder().fit(all_airlines)
    le_origin = LabelEncoder().fit(all_airports)
    le_dest = LabelEncoder().fit(all_airports)

    # Safe transform with fallback
    def safe_enc(le, val):
        try:
            return int(le.transform([val])[0])
        except ValueError:
            return 0

    raw = {
        "airline_enc": safe_enc(le_airline, airline),
        "origin_enc": safe_enc(le_origin, origin),
        "dest_enc": safe_enc(le_dest, dest),
        "dep_hour": dep_hour,
        "dep_month": dep_month,
        "dep_dayofweek": dep_day % 7,
        "distance": distance,
        "scheduled_elapsed": scheduled_elapsed,
        "carrier_delay_hist": round(1 - AIRLINE_RELIABILITY.get(airline, 0.72), 3),
        "airport_congestion": DELAY_PRONE_AIRPORTS.get(origin, 0.15),
        "is_weekend": int(dep_day % 7 >= 5),
        "is_peak_hour": int(((dep_hour >= 7) and (dep_hour <= 9)) or
                            ((dep_hour >= 17) and (dep_hour <= 20))),
        "is_holiday_season": int(dep_month == 12 or dep_month == 1 or
                                 (dep_month == 11 and dep_day >= 20)),
        "weather_risk": {1: 0.8, 2: 0.75, 3: 0.4, 4: 0.3, 5: 0.25, 6: 0.2,
                         7: 0.22, 8: 0.25, 9: 0.3, 10: 0.35, 11: 0.55, 12: 0.75
                         }.get(dep_month, 0.3),
        "route_delay_rate": 0.22,  # Global average (no historical route data for live pred)
        "dep_hour_sin": np.sin(2 * np.pi * dep_hour / 24),
        "dep_hour_cos": np.cos(2 * np.pi * dep_hour / 24),
        "month_sin": np.sin(2 * np.pi * dep_month / 12),
        "month_cos": np.cos(2 * np.pi * dep_month / 12),
    }

    # Build vector in exact feature order
    vec = np.array([[raw.get(f, 0) for f in feature_names]])
    vec_scaled = scaler.transform(vec)

    return vec_scaled, raw


# ── Prediction + Explainability ────────────────────────────────────────────────
def predict_delay(
    airline: str, origin: str, dest: str,
    dep_month: int, dep_day: int, dep_hour: int,
    distance: int, scheduled_elapsed: int,
) -> dict:
    """
    Full prediction pipeline. Returns:
      - delay_probability (0-1)
      - is_delayed (bool)
      - risk_level: "Low" / "Medium" / "High" / "Critical"
      - feature_contributions: dict of human-readable factor impacts
      - recommendations: list of strings
    """
    model, scaler, metadata = load_artifacts()
    feature_names = metadata["feature_names"]

    vec_scaled, raw = build_feature_vector(
        airline, origin, dest, dep_month, dep_day, dep_hour,
        distance, scheduled_elapsed, feature_names, scaler
    )

    prob = float(model.predict_proba(vec_scaled)[0][1])

    # ── Risk Level ──
    if prob < 0.30:
        risk_level = "Low"
        risk_color = "#22c55e"
    elif prob < 0.55:
        risk_level = "Medium"
        risk_color = "#f59e0b"
    elif prob < 0.75:
        risk_level = "High"
        risk_color = "#f97316"
    else:
        risk_level = "Critical"
        risk_color = "#ef4444"

    # ── Approximate Feature Contributions (manual SHAP proxy) ──
    contributions = compute_feature_contributions(raw, prob, airline, origin, dep_month, dep_hour)

    # ── Recommendations ──
    recs = generate_recommendations(raw, prob, airline, origin, dep_month, dep_hour)

    return {
        "delay_probability": round(prob, 4),
        "delay_percent": round(prob * 100, 1),
        "is_delayed": prob >= 0.50,
        "risk_level": risk_level,
        "risk_color": risk_color,
        "raw_features": raw,
        "feature_contributions": contributions,
        "recommendations": recs,
        "model_used": metadata["best_model"],
    }


def compute_feature_contributions(raw: dict, prob: float, airline: str,
                                   origin: str, dep_month: int, dep_hour: int) -> list:
    """
    Produces human-readable feature contribution breakdown.
    Uses domain rules as a proxy for SHAP values (no external SHAP dependency).
    Returns list of {factor, impact, direction, value}.
    """
    contribs = []
    base = 0.20  # baseline delay rate

    # Airline factor
    carrier_impact = raw["carrier_delay_hist"] * 0.30
    contribs.append({
        "factor": f"Airline ({airline})",
        "impact": round(carrier_impact, 3),
        "direction": "↑" if carrier_impact > 0.06 else "↓",
        "value": f"{raw['carrier_delay_hist']:.0%} historical delay rate",
    })

    # Airport congestion
    congestion_impact = raw["airport_congestion"] * 0.35
    contribs.append({
        "factor": f"Origin Airport ({origin})",
        "impact": round(congestion_impact, 3),
        "direction": "↑" if raw["airport_congestion"] > 0.20 else "→",
        "value": f"Congestion score: {raw['airport_congestion']:.2f}",
    })

    # Weather risk
    weather_impact = raw["weather_risk"] * 0.25
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                   7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    contribs.append({
        "factor": f"Weather Risk ({month_names[dep_month]})",
        "impact": round(weather_impact, 3),
        "direction": "↑" if raw["weather_risk"] > 0.4 else "↓",
        "value": f"Seasonal risk: {raw['weather_risk']:.2f}",
    })

    # Peak hour
    if raw["is_peak_hour"]:
        contribs.append({
            "factor": f"Departure Time ({dep_hour:02d}:00)",
            "impact": 0.10,
            "direction": "↑",
            "value": "Peak hour congestion window",
        })
    else:
        contribs.append({
            "factor": f"Departure Time ({dep_hour:02d}:00)",
            "impact": -0.05,
            "direction": "↓",
            "value": "Off-peak departure",
        })

    # Holiday season
    if raw["is_holiday_season"]:
        contribs.append({
            "factor": "Holiday Season",
            "impact": 0.07,
            "direction": "↑",
            "value": "High-traffic holiday period",
        })

    # Sort by absolute impact
    contribs.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return contribs


def generate_recommendations(raw: dict, prob: float, airline: str,
    origin: str, dep_month: int, dep_hour: int) -> list:
    """Generates actionable delay avoidance recommendations."""
    recs = []

    if raw["is_peak_hour"]:
        recs.append("✈️ Consider rebooking to an off-peak departure (before 7am or 10am–4pm) to reduce congestion risk.")

    if raw["weather_risk"] > 0.5:
        recs.append("🌨️ High weather risk this month. Monitor NOAA forecasts 24h before departure.")

    if raw["airport_congestion"] > 0.25:
        recs.append(f"🚦 {origin} has elevated congestion. Arrive 30 min earlier than usual for check-in.")

    if raw["carrier_delay_hist"] > 0.30:
        recs.append(f"⚠️ {airline} has a higher-than-average delay rate. Consider booking a backup plan.")

    if raw["is_holiday_season"]:
        recs.append("🎄 Holiday season travel: book early-morning flights which have the lowest delay rates.")

    if prob < 0.25:
        recs.append("✅ This flight has a low delay probability. Standard check-in timing should be fine.")

    if not recs:
        recs.append("📊 Moderate delay risk. Standard precautions apply — check airline app for updates.")

    return recs