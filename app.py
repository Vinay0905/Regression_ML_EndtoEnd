import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import os
from pathlib import Path

# Import Supabase client utility
from src.utils.supabase_client import get_supabase_client

# ============================
# Config
# ============================
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")
STORAGE_BUCKET = "housing-data"

# ============================
# Helpers
# ============================
def load_from_supabase(remote_path: str, local_path: Path):
    """Download from Supabase Storage with fallback checks. No UI side-effects."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        supabase = get_supabase_client()
        
        # 1. Try the direct path
        try:
            res = supabase.storage.from_(STORAGE_BUCKET).download(remote_path)
        except Exception:
            # 2. Try root fallback
            filename_only = remote_path.split("/")[-1]
            res = supabase.storage.from_(STORAGE_BUCKET).download(filename_only)
        
        with open(local_path, "wb") as f:
            f.write(res)
    return str(local_path)

@st.cache_data(show_spinner=False)
def load_data():
    """Fetch holdout data from Supabase and load into memory once."""
    engineered_local = "data/processed/feature_engineered_holdout.csv"
    meta_local = "data/processed/cleaning_holdout.csv"
    
    # These calls are fast if file exists on disk
    holdout_path = load_from_supabase("processed/feature_engineered_holdout.csv", Path(engineered_local))
    meta_path = load_from_supabase("processed/cleaning_holdout.csv", Path(meta_local))

    if not holdout_path or not meta_path:
        return None, None

    # Load CSVs - using simplified dtypes could save memory if needed later
    fe = pd.read_csv(holdout_path)
    meta = pd.read_csv(meta_path, parse_dates=["date"])

    # Align Data
    min_len = min(len(fe), len(meta))
    fe = fe.iloc[:min_len].copy()
    meta = meta.iloc[:min_len].copy()

    # Prepare small display dataframe to save RAM
    disp = pd.DataFrame({
        "date": meta["date"],
        "region": meta["city_full"],
        "year": meta["date"].dt.year,
        "month": meta["date"].dt.month,
        "actual_price": fe["price"]
    })

    return fe, disp

# ============================
# UI Setup
# ============================
st.set_page_config(page_title="Housing Prediction Dashboard", layout="wide")

# Main Title
st.title("🏠 Housing Price Prediction — Holdout Explorer")

# Data loading with a single controlled spinner
with st.spinner("🚀 Initializing application and downloading datasets from Supabase..."):
    fe_df, disp_df = load_data()

if fe_df is None or disp_df.empty:
    st.error("❌ Critical Error: Data files could not be loaded. Please check Supabase Storage and Environment Variables.")
    st.stop()

# ============================
# Sidebar/Filters
# ============================
years = sorted(disp_df["year"].unique())
months = list(range(1, 13))
regions = ["All"] + sorted(disp_df["region"].dropna().unique())

col1, col2, col3 = st.columns(3)
with col1:
    year = st.selectbox("Select Year", years, index=0)
with col2:
    month = st.selectbox("Select Month", months, index=0)
with col3:
    region = st.selectbox("Select Region", regions, index=0)

# ============================
# Execution logic
# ============================
if st.button("Show Predictions 🚀"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning(f"No data found for {year}-{month:02d} in {region}.")
    else:
        st.write(f"📅 Running predictions for **{year}-{month:02d}** | Region: **{region}**")
        
        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            with st.spinner("Calculating predictions via API..."):
                resp = requests.post(API_URL, json=payload, timeout=60)
                resp.raise_for_status()
                out = resp.json()
            
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            # Build View
            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)
            if actuals:
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(float)

            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            
            st.subheader("Results")
            st.dataframe(view.reset_index(drop=True), use_container_width=True)

            m1, m2 = st.columns(2)
            m1.metric("MAE", f"${mae:,.0f}")
            m2.metric("RMSE", f"${rmse:,.0f}")

            # Plot
            st.divider()
            # Minimal aggregation for chart
            monthly_data = disp_df[disp_df["year"] == year].copy()
            if region != "All":
                monthly_data = monthly_data[monthly_data["region"] == region]
            
            fig = px.line(monthly_data.groupby("month")[["actual_price"]].mean().reset_index(), 
                         x="month", y="actual_price", markers=True, title=f"Price Trend {year}")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
else:
    st.info("Choose filters and click **Show Predictions** to begin.")
