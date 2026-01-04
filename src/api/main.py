# Goal: Create a FastAPI app to serve your trained ML model via HTTP.
# Now powered by Supabase for Storage (Model) and Database (Logs).

from fastapi import FastAPI, BackgroundTasks
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import os
from datetime import datetime
import json

# Import custom modules
from src.inference_pipeline.inference import predict
from src.utils.supabase_client import get_supabase_client

# ----------------------------
# Config
# ----------------------------
# Buckets
STORAGE_BUCKET = "housing-data"
MODELS_PATH_REMOTE = "models/xgb_best_model.pkl"
DATA_PATH_REMOTE = "data/processed/feature_engineered_train.csv"

# Local Paths
MODEL_PATH = Path("models/xgb_best_model.pkl")
TRAIN_FE_PATH = Path("data/processed/feature_engineered_train.csv")

# ----------------------------
# Helpers
# ----------------------------
def load_from_supabase(remote_path: str, local_path: Path):
    """Download file from Supabase Storage if not cached."""
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        print(f"📥 Downloading {remote_path} from Supabase Storage...")
        
        supabase = get_supabase_client()
        # Download binary
        res = supabase.storage.from_(STORAGE_BUCKET).download(remote_path)
        
        with open(local_path, "wb") as f:
            f.write(res)
        print(f"✅ Saved to {local_path}")
    return local_path

async def log_prediction_to_db(inputs: List[dict], predictions: List[float]):
    """Asynchronously log inputs and predictions to Supabase DB."""
    try:
        supabase = get_supabase_client()
        
        # Prepare rows for insertion
        rows = []
        for i, pred in enumerate(predictions):
            row = {
                "input_data": json.dumps(inputs[i]),  # Store input JSON
                "prediction": pred,
                "created_at": datetime.utcnow().isoformat()
            }
            rows.append(row)
            
        supabase.table("predictions").insert(rows).execute()
        print(f"📝 Logged {len(rows)} predictions to Supabase.")
        
    except Exception as e:
        print(f"⚠️ Failed to log to Supabase: {e}")

# ----------------------------
# Startup Logic
# ----------------------------
# We try to download artifacts on import, but robustly handle failures if env vars are missing during build.
TRAIN_FEATURE_COLUMNS = None

try:
    load_from_supabase(MODELS_PATH_REMOTE, MODEL_PATH)
    load_from_supabase(DATA_PATH_REMOTE, TRAIN_FE_PATH)
    
    if TRAIN_FE_PATH.exists():
        _train_cols = pd.read_csv(TRAIN_FE_PATH, nrows=1)
        TRAIN_FEATURE_COLUMNS = [c for c in _train_cols.columns if c != "price"]
except Exception as e:
    print(f"⚠️ Startup Warning: Could not download artifacts. {e}")

# ----------------------------
# App Definition
# ----------------------------
app = FastAPI(title="Housing Regression API (Supabase Edition)")

@app.get("/")
def root():
    return {"message": "Housing Regression API is running 🚀 (Supabase Enabled)"}

@app.get("/health")
def health():
    status: Dict[str, Any] = {"model_path": str(MODEL_PATH)}
    
    # Check Model
    if MODEL_PATH.exists():
        status["model_status"] = "present"
    else:
        status["model_status"] = "missing"

    # Check Supabase Connection
    try:
        supabase = get_supabase_client()
        # Lightweight check: List buckets
        supabase.storage.list_buckets()
        status["supabase_connection"] = "healthy"
    except Exception as e:
        status["supabase_connection"] = f"unhealthy: {str(e)}"

    return status

@app.post("/predict")
async def predict_endpoint(data: List[dict], background_tasks: BackgroundTasks):
    if not MODEL_PATH.exists():
        return {"error": "Model not found. Please ensure it is uploaded to Supabase Storage."}

    df = pd.DataFrame(data)
    if df.empty:
        return {"error": "No data provided"}

    # Run Inference
    preds_df = predict(df, model_path=MODEL_PATH)
    predictions_list = preds_df["predicted_price"].astype(float).tolist()

    # Log to Supabase in background (doesn't block response)
    background_tasks.add_task(log_prediction_to_db, data, predictions_list)

    resp = {"predictions": predictions_list}
    if "actual_price" in preds_df.columns:
        resp["actuals"] = preds_df["actual_price"].astype(float).tolist()

    return resp
