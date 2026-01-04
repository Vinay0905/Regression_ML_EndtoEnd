"""
Inference pipeline for Housing Regression MLE.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from joblib import load
import os

# Import preprocessing + feature engineering helpers
from src.feature_pipeline.preprocess import clean_and_merge, drop_duplicates, remove_outliers
from src.feature_pipeline.feature_engineering import add_date_features, drop_unused_columns

# ----------------------------
# Default paths
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = PROJECT_ROOT / "models" / "xgb_best_model.pkl"
DEFAULT_FREQ_ENCODER = PROJECT_ROOT / "models" / "freq_encoder.pkl"
DEFAULT_TARGET_ENCODER = PROJECT_ROOT / "models" / "target_encoder.pkl"
TRAIN_FE_PATH = PROJECT_ROOT / "data" / "processed" / "feature_engineered_train.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "predictions.csv"

def get_training_columns():
    """Dynamically load training columns from disk to avoid startup race conditions."""
    if TRAIN_FE_PATH.exists():
        try:
            _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
            return [c for c in _train_cols.columns if c != "price"]
        except Exception:
            return None
    return None

def predict(
    input_df: pd.DataFrame,
    model_path: Path | str = DEFAULT_MODEL,
    freq_encoder_path: Path | str = DEFAULT_FREQ_ENCODER,
    target_encoder_path: Path | str = DEFAULT_TARGET_ENCODER,
) -> pd.DataFrame:
    # Step 1: Preprocess raw input
    df = clean_and_merge(input_df)
    
    # Preserve lat/lng if they exist in input for model alignment
    preserved_cols = {}
    for col in ["lat", "lng"]:
        if col in df.columns:
            preserved_cols[col] = df[col]

    df = drop_duplicates(df)
    df = remove_outliers(df)

    # Step 2: Feature engineering
    if "date" in df.columns:
        df = add_date_features(df)

    # Step 3: Encodings
    if Path(freq_encoder_path).exists() and "zipcode" in df.columns:
        freq_map = load(freq_encoder_path)
        df["zipcode_freq"] = df["zipcode"].map(freq_map).fillna(0)
        df = df.drop(columns=["zipcode"], errors="ignore")

    if Path(target_encoder_path).exists() and "city_full" in df.columns:
        target_encoder = load(target_encoder_path)
        df["city_encoded"] = target_encoder.transform(df["city_full"])
        # Do not drop city_full yet if we need it for alignment, 
        # but the log showed 'city_encoded' was expected.

    # Step 4: Drop leakage columns
    df, _ = drop_unused_columns(df.copy(), df.copy())

    # Restore preserved lat/lng if missing
    for col, values in preserved_cols.items():
        if col not in df.columns:
            df[col] = values

    # Step 5: Separate actuals if present
    y_true = None
    if "price" in df.columns:
        y_true = df["price"].tolist()
        df = df.drop(columns=["price"])

    # Step 6: Align columns with training schema (Dynamic check)
    train_cols = get_training_columns()
    if train_cols is not None:
        # Reindex ensures all expected columns exist (fills with 0 if missing)
        # and ensures they are in the EXACT order the Booster expects.
        df = df.reindex(columns=train_cols, fill_value=0)

    # Step 7: Load model & predict
    model = load(model_path)
    preds = model.predict(df)

    # Build output
    out = df.copy()
    out["predicted_price"] = preds
    if y_true is not None:
        out["actual_price"] = y_true

    return out
