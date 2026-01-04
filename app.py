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
    """Download from Supabase Storage if not already cached locally."""
    local_path = Path(local_path)
    if not local_path.exists():
        os.makedirs(local_path.parent, exist_ok=True)
        try:
            supabase = get_supabase_client()
            res = supabase.storage.from_(STORAGE_BUCKET).download(remote_path)
            with open(local_path, "wb") as f:
                f.write(res)
            st.success(f"✅ Downloaded {remote_path}")
        except Exception as e:
            st.error(f"❌ Failed to download {remote_path} from Supabase: {e}")
            return None
    return str(local_path)

# Paths (ensure available locally by fetching from Supabase if missing)
HOLDOUT_ENGINEERED_PATH = load_from_supabase(
    "processed/feature_engineered_holdout.csv",
    "data/processed/feature_engineered_holdout.csv"
)
HOLDOUT_META_PATH = load_from_supabase(
    "processed/cleaning_holdout.csv",
    "data/processed/cleaning_holdout.csv"
)

# ============================
# Data loading
# ============================
@st.cache_data
def load_data():
    if not HOLDOUT_ENGINEERED_PATH or not HOLDOUT_META_PATH:
        st.error("Missing data files. Ensure they are uploaded to Supabase Storage.")
        return pd.DataFrame(), pd.DataFrame()

    fe = pd.read_csv(HOLDOUT_ENGINEERED_PATH)
    meta = pd.read_csv(HOLDOUT_META_PATH, parse_dates=["date"])[["date", "city_full"]]

    if len(fe) != len(meta):
        st.warning("⚠️ Engineered and meta holdout lengths differ. Aligning by index.")
        min_len = min(len(fe), len(meta))
        fe = fe.iloc[:min_len].copy()
        meta = meta.iloc[:min_len].copy()

    disp = pd.DataFrame(index=fe.index)
    disp["date"] = meta["date"]
    disp["region"] = meta["city_full"]
    disp["year"] = disp["date"].dt.year
    disp["month"] = disp["date"].dt.month
    disp["actual_price"] = fe["price"]

    return fe, disp

fe_df, disp_df = load_data()

# Check if data loaded successfully
if disp_df.empty:
    st.stop()

# ============================
# UI
# ============================
st.set_page_config(page_title="Housing Prediction Dashboard", layout="wide")
st.title("🏠 Housing Price Prediction — Holdout Explorer")

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

if st.button("Show Predictions 🚀"):
    mask = (disp_df["year"] == year) & (disp_df["month"] == month)
    if region != "All":
        mask &= (disp_df["region"] == region)

    idx = disp_df.index[mask]

    if len(idx) == 0:
        st.warning("No data found for these filters.")
    else:
        st.write(f"📅 Running predictions for **{year}-{month:02d}** | Region: **{region}**")

        payload = fe_df.loc[idx].to_dict(orient="records")

        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            out = resp.json()
            preds = out.get("predictions", [])
            actuals = out.get("actuals", None)

            view = disp_df.loc[idx, ["date", "region", "actual_price"]].copy()
            view = view.sort_values("date")
            view["prediction"] = pd.Series(preds, index=view.index).astype(float)

            if actuals is not None and len(actuals) == len(view):
                view["actual_price"] = pd.Series(actuals, index=view.index).astype(float)

            # Metrics
            mae = (view["prediction"] - view["actual_price"]).abs().mean()
            rmse = ((view["prediction"] - view["actual_price"]) ** 2).mean() ** 0.5
            avg_pct_error = ((view["prediction"] - view["actual_price"]).abs() / view["actual_price"]).mean() * 100

            st.subheader("Predictions vs Actuals")
            st.dataframe(
                view[["date", "region", "actual_price", "prediction"]].reset_index(drop=True),
                use_container_width=True
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("MAE", f"{mae:,.0f}")
            with c2:
                st.metric("RMSE", f"{rmse:,.0f}")
            with c3:
                st.metric("Avg % Error", f"{avg_pct_error:.2f}%")

            # Yearly Trend
            st.divider()
            if region == "All":
                yearly_data = disp_df[disp_df["year"] == year].copy()
            else:
                yearly_data = disp_df[(disp_df["year"] == year) & (disp_df["region"] == region)].copy()
            
            idx_yearly = yearly_data.index
            payload_yearly = fe_df.loc[idx_yearly].to_dict(orient="records")
            
            resp_yearly = requests.post(API_URL, json=payload_yearly, timeout=60)
            resp_yearly.raise_for_status()
            preds_yearly = resp_yearly.json().get("predictions", [])
            
            yearly_data["prediction"] = pd.Series(preds_yearly, index=yearly_data.index).astype(float)
            monthly_avg = yearly_data.groupby("month")[["actual_price", "prediction"]].mean().reset_index()

            fig = px.line(
                monthly_avg,
                x="month",
                y=["actual_price", "prediction"],
                markers=True,
                labels={"value": "Price", "month": "Month"},
                title=f"Yearly Trend — {year}{'' if region=='All' else f' — {region}'}"
            )
            
            fig.add_vrect(
                x0=month - 0.5, x1=month + 0.5,
                fillcolor="red", opacity=0.1, layer="below", line_width=0,
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"API call failed: {e}")
            st.exception(e)
else:
    st.info("Choose filters and click **Show Predictions** to compute.")
