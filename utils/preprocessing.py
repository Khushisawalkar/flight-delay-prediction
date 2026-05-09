"""
preprocessing.py — Data ingestion, cleaning, and feature engineering pipeline.

Handles both synthetic dataset generation and real CSV uploads.
Produces a clean, ML-ready feature matrix with rich derived features.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
AIRLINES = ["Delta", "United", "American", "Southwest", "JetBlue",
            "Spirit", "Frontier", "Alaska", "Hawaiian", "Allegiant"]

AIRPORTS = ["ATL", "LAX", "ORD", "DFW", "DEN", "JFK", "SFO", "SEA",
            "LAS", "MCO", "MIA", "PHX", "IAH", "CLT", "BOS"]

# Airports known for weather/congestion delays (for realistic simulation)
DELAY_PRONE_AIRPORTS = {"ORD": 0.35, "JFK": 0.30, "SFO": 0.28, "BOS": 0.32, "DEN": 0.25}

# Airline historical on-time performance (lower = more delays)
AIRLINE_RELIABILITY = {
    "Delta": 0.82, "Alaska": 0.80, "Hawaiian": 0.85,
    "United": 0.72, "American": 0.70, "Southwest": 0.75,
    "JetBlue": 0.68, "Spirit": 0.55, "Frontier": 0.58, "Allegiant": 0.62
}

TARGET_COL = "is_delayed"
FEATURE_COLS = [
    "airline_enc", "origin_enc", "dest_enc",
    "dep_hour", "dep_month", "dep_dayofweek",
    "distance", "scheduled_elapsed",
    "carrier_delay_hist", "airport_congestion",
    "is_weekend", "is_peak_hour", "is_holiday_season",
    "weather_risk", "route_delay_rate",
    "dep_hour_sin", "dep_hour_cos",
    "month_sin", "month_cos",
]


# ── Synthetic Dataset Generator ────────────────────────────────────────────────
def generate_synthetic_dataset(n_rows: int = 15000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic airline delay dataset.
    Uses domain knowledge to inject realistic delay patterns:
      - Airports with known congestion issues have higher base delay rates
      - Weather risk is seasonally correlated (winter months = higher risk)
      - Peak hours (7-9am, 5-8pm) have higher delay probability
      - Delay cascades: late aircraft propagate through the day
    """
    rng = np.random.default_rng(random_state)

    n = n_rows
    airlines = rng.choice(AIRLINES, n)
    origins = rng.choice(AIRPORTS, n)
    dests = rng.choice(AIRPORTS, n)

    # Ensure origin != dest
    mask = origins == dests
    dests[mask] = rng.choice(AIRPORTS, mask.sum())

    months = rng.integers(1, 13, n)
    days = rng.integers(1, 29, n)
    dep_hours = rng.choice(range(5, 24), n,
                           p=[0.06, 0.10, 0.12, 0.10, 0.08, 0.06,
                              0.05, 0.04, 0.05, 0.06, 0.07, 0.06,
                              0.05, 0.04, 0.04, 0.04, 0.04, 0.03, 0.01])
    distances = rng.integers(100, 3000, n)
    elapsed = (distances / 500 * 60 + rng.normal(0, 15, n)).astype(int).clip(45, 600)

    # --- Feature-driven delay probability ---
    prob = np.full(n, 0.20)  # Base rate ~20%

    # Airline factor
    airline_reliability = np.array([AIRLINE_RELIABILITY[a] for a in airlines])
    prob += (1 - airline_reliability) * 0.5

    # Airport congestion
    airport_congestion = np.array([DELAY_PRONE_AIRPORTS.get(o, 0.15) for o in origins])
    prob += airport_congestion * 0.4

    # Peak hours: 7-9 AM and 5-8 PM
    is_peak = ((dep_hours >= 7) & (dep_hours <= 9)) | ((dep_hours >= 17) & (dep_hours <= 20))
    prob += is_peak * 0.12

    # Late-day cascades: delays compound through the day
    prob += (dep_hours > 18) * 0.10

    # Winter months (Dec, Jan, Feb) = weather risk
    weather_risk_by_month = {12: 0.18, 1: 0.20, 2: 0.16, 3: 0.08,
                              6: 0.05, 7: 0.06, 8: 0.07}
    weather_risk = np.array([weather_risk_by_month.get(m, 0.05) for m in months])
    prob += weather_risk * 0.6

    # Holiday season (late Nov, Dec)
    is_holiday = ((months == 11) & (days >= 20)) | (months == 12)
    prob += is_holiday * 0.08

    # Long haul = more variance
    prob += (distances > 2000) * 0.05

    prob = np.clip(prob, 0.05, 0.92)
    is_delayed = rng.binomial(1, prob).astype(int)

    # Delay minutes (only for delayed flights, for analytics)
    delay_minutes = np.zeros(n)
    delayed_mask = is_delayed == 1
    delay_minutes[delayed_mask] = rng.exponential(35, delayed_mask.sum()).clip(15, 300)

    df = pd.DataFrame({
        "airline": airlines,
        "origin": origins,
        "dest": dests,
        "dep_month": months,
        "dep_day": days,
        "dep_hour": dep_hours,
        "distance": distances,
        "scheduled_elapsed": elapsed,
        "delay_minutes": delay_minutes.astype(int),
        "is_delayed": is_delayed,
    })

    return df


# ── Feature Engineering ────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derives ML features from raw flight data.
    Adds:
      - Temporal cyclical encodings (sin/cos for hour, month)
      - Historical delay rates per airline and airport
      - Binary flags: weekend, peak hour, holiday season
      - Weather risk proxy
      - Route-level historical delay rate
    """
    df = df.copy()

    # ── Encoders ──
    le_airline = LabelEncoder()
    le_origin = LabelEncoder()
    le_dest = LabelEncoder()

    df["airline_enc"] = le_airline.fit_transform(df["airline"])
    df["origin_enc"] = le_origin.fit_transform(df["origin"])
    df["dest_enc"] = le_dest.fit_transform(df["dest"])

    # ── Historical delay rate per carrier ──
    df["carrier_delay_hist"] = df["airline"].map(
        {a: round(1 - r, 3) for a, r in AIRLINE_RELIABILITY.items()}
    ).fillna(0.25)

    # ── Airport congestion score ──
    df["airport_congestion"] = df["origin"].map(DELAY_PRONE_AIRPORTS).fillna(0.15)

    # ── Day-of-week (0=Mon) — estimate from day/month ──
    df["dep_dayofweek"] = (df["dep_day"] % 7).astype(int)

    # ── Binary flags ──
    df["is_weekend"] = (df["dep_dayofweek"] >= 5).astype(int)
    df["is_peak_hour"] = (
        ((df["dep_hour"] >= 7) & (df["dep_hour"] <= 9)) |
        ((df["dep_hour"] >= 17) & (df["dep_hour"] <= 20))
    ).astype(int)
    df["is_holiday_season"] = (
        (df["dep_month"] == 12) | (df["dep_month"] == 1) |
        ((df["dep_month"] == 11) & (df["dep_day"] >= 20))
    ).astype(int)

    # ── Weather risk proxy (seasonal) ──
    weather_map = {1: 0.8, 2: 0.75, 3: 0.4, 4: 0.3, 5: 0.25, 6: 0.2,
                   7: 0.22, 8: 0.25, 9: 0.3, 10: 0.35, 11: 0.55, 12: 0.75}
    df["weather_risk"] = df["dep_month"].map(weather_map).fillna(0.3)

    # ── Route-level historical delay rate ──
    df["route"] = df["origin"] + "_" + df["dest"]
    route_rates = df.groupby("route")["is_delayed"].mean().to_dict()
    df["route_delay_rate"] = df["route"].map(route_rates).fillna(0.2)

    # ── Cyclical encodings (hour) ──
    df["dep_hour_sin"] = np.sin(2 * np.pi * df["dep_hour"] / 24)
    df["dep_hour_cos"] = np.cos(2 * np.pi * df["dep_hour"] / 24)

    # ── Cyclical encodings (month) ──
    df["month_sin"] = np.sin(2 * np.pi * df["dep_month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["dep_month"] / 12)

    return df


# ── Split & Scale ──────────────────────────────────────────────────────────────
def prepare_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Returns X_train, X_test, y_train, y_test, scaler, feature_names.
    Applies StandardScaler fitted on train set only.
    """
    df_feat = engineer_features(df)

    # Keep only available FEATURE_COLS
    available_feats = [c for c in FEATURE_COLS if c in df_feat.columns]
    X = df_feat[available_feats].fillna(0).values
    y = df_feat[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train_sc, X_test_sc, y_train, y_test, scaler, available_feats


# ── CSV Upload Normalizer ──────────────────────────────────────────────────────
COLUMN_ALIASES = {
    "carrier": "airline", "fl_num": "flight_num",
    "origin_airport": "origin", "dest_airport": "dest",
    "departure_delay": "delay_minutes", "dep_delay": "delay_minutes",
    "crs_dep_time": "dep_hour", "month": "dep_month",
    "day_of_month": "dep_day", "day_of_week": "dep_dayofweek",
    "crs_elapsed_time": "scheduled_elapsed",
    "arr_delay": "delay_minutes",
}

REQUIRED_COLS = {"airline", "origin", "dest", "dep_month", "dep_hour", "distance"}


def normalize_uploaded_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes user-uploaded CSVs to our internal schema.
    Handles common column name variations from public aviation datasets
    (BTS, Kaggle airline delay datasets, etc.)
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Apply aliases
    df.rename(columns=COLUMN_ALIASES, inplace=True)

    # Parse dep_hour from HHMM format if needed (e.g. 1430 → 14)
    if "dep_hour" in df.columns and df["dep_hour"].max() > 24:
        df["dep_hour"] = (df["dep_hour"] // 100).clip(0, 23)

    # Derive is_delayed from delay_minutes if present
    if "is_delayed" not in df.columns:
        if "delay_minutes" in df.columns:
            df["is_delayed"] = (df["delay_minutes"] >= 15).astype(int)
        else:
            df["is_delayed"] = 0  # Unknown — will use for analytics only

    # Ensure numeric
    numeric_cols = ["dep_month", "dep_hour", "distance", "scheduled_elapsed",
                    "delay_minutes", "is_delayed", "dep_day"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(REQUIRED_COLS.intersection(set(df.columns))))

    # Fill missing optionals
    df["dep_month"] = df.get("dep_month", pd.Series([6] * len(df))).fillna(6)
    df["dep_day"] = df.get("dep_day", pd.Series([15] * len(df))).fillna(15)
    df["dep_hour"] = df.get("dep_hour", pd.Series([12] * len(df))).fillna(12)
    df["distance"] = df.get("distance", pd.Series([800] * len(df))).fillna(800)
    df["scheduled_elapsed"] = df.get("scheduled_elapsed",
                                     pd.Series([120] * len(df))).fillna(120)
    df["delay_minutes"] = df.get("delay_minutes", pd.Series([0] * len(df))).fillna(0)

    return df.reset_index(drop=True)