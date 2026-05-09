"""
train_model.py — ML training pipeline.

Trains and evaluates 3 algorithms:
  1. Random Forest
  2. XGBoost (GradientBoostingClassifier as fallback if xgboost not installed)
  3. Logistic Regression

Saves models + metadata to models/ directory.
Run standalone: python train_model.py
"""

import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score,
    confusion_matrix, roc_curve, classification_report
)
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from utils.preprocessing import (
    generate_synthetic_dataset, prepare_train_test, FEATURE_COLS
)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# ── Model Definitions ──────────────────────────────────────────────────────────
def get_model_configs():
    """
    Returns dict of {name: (model_instance, params_label)}.
    Uses tuned hyperparameters from domain knowledge + lightweight grid search.
    """
    configs = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        configs["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=2,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    else:
        configs["XGBoost (GB)"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

    return configs


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str, feature_names: list) -> dict:
    """
    Computes full evaluation suite: accuracy, F1, AUC, confusion matrix,
    ROC curve data, feature importances, and classification report.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    # ROC curve data (serializable)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Feature importances
    importances = {}
    if hasattr(model, "feature_importances_"):
        for fname, imp in zip(feature_names, model.feature_importances_):
            importances[fname] = round(float(imp), 5)
    elif hasattr(model, "coef_"):
        for fname, coef in zip(feature_names, model.coef_[0]):
            importances[fname] = round(float(abs(coef)), 5)

    return {
        "model_name": model_name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(auc), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "confusion_matrix": cm,
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
        "feature_importances": importances,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }


# ── Main Training Pipeline ─────────────────────────────────────────────────────
def train_all_models(df: pd.DataFrame = None, verbose: bool = True) -> dict:
    """
    Full training pipeline. Accepts optional DataFrame; generates synthetic data if None.
    Returns dict with all model results + metadata.
    """
    if df is None:
        if verbose:
            print("📊 Generating synthetic dataset (15,000 flights)...")
        df = generate_synthetic_dataset(n_rows=15000)

    if verbose:
        print(f"✅ Dataset: {len(df):,} rows | Delay rate: {df['is_delayed'].mean():.1%}")

    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_train_test(df)

    if verbose:
        print(f"🔀 Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"📐 Features: {feature_names}\n")

    model_configs = get_model_configs()
    results = {}
    trained_models = {}

    for name, model in model_configs.items():
        if verbose:
            print(f"🤖 Training {name}...")

        start = time.time()
        model.fit(X_train, y_train)
        train_time = round(time.time() - start, 2)

        metrics = evaluate_model(model, X_test, y_test, name, feature_names)
        metrics["train_time_sec"] = train_time
        results[name] = metrics
        trained_models[name] = model

        if verbose:
            print(f"   ✓ Acc={metrics['accuracy']:.3f} | F1={metrics['f1_score']:.3f} "
                  f"| AUC={metrics['roc_auc']:.3f} | {train_time}s")

    # ── Save artifacts ──
    # Best model by AUC
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = trained_models[best_name]

    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Save all models
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        with open(MODELS_DIR / f"{safe_name}.pkl", "wb") as f:
            pickle.dump(model, f)

    # Save metadata
    metadata = {
        "best_model": best_name,
        "feature_names": feature_names,
        "dataset_size": len(df),
        "delay_rate": round(float(df["is_delayed"].mean()), 4),
        "trained_at": pd.Timestamp.now().isoformat(),
        "results": results,
    }

    with open(MODELS_DIR / "metadata.json", "w") as f:
        # Convert numpy types for JSON serialization
        json.dump(metadata, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    if verbose:
        print(f"\n🏆 Best model: {best_name} (AUC={results[best_name]['roc_auc']:.3f})")
        print(f"💾 Models saved to {MODELS_DIR}/")

    return metadata


if __name__ == "__main__":
    train_all_models(verbose=True)