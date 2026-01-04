## Housing ML end2end Project

## Vinay's Project for complete understanding of workflows and working and complete deployment.

## Project Overview

Housing Regression MLE is an end-to-end machine learning pipeline for predicting housing prices using XGBoost. The project follows ML engineering best practices with modular pipelines, experiment tracking via MLflow, containerization, **cloud deployment on Google Cloud**, and comprehensive testing. The system includes both a REST API and a Streamlit dashboard for interactive predictions.

## Architecture

The codebase is organized into distinct pipelines following the flow:
`Load → Preprocess → Feature Engineering → Train → Tune → Evaluate → Inference → Serve`

### Core Modules

- **`src/feature_pipeline/`**: Data loading, preprocessing, and feature engineering

  - `load.py`: Time-aware data splitting (train <2020, eval 2020-21, holdout ≥2022)
  - `preprocess.py`: City normalization, deduplication, outlier removal
  - `feature_engineering.py`: Date features, frequency encoding (zipcode), target encoding (city_full)

- **`src/training_pipeline/`**: Model training and hyperparameter optimization

  - `train.py`: Baseline XGBoost training with configurable parameters
  - `tune.py`: Optuna-based hyperparameter tuning with MLflow integration
  - `eval.py`: Model evaluation and metrics calculation

- **`src/inference_pipeline/`**: Production inference

  - `inference.py`: Applies same preprocessing/encoding transformations using saved encoders

- **`src/api/`**: FastAPI web service
  - `main.py`: REST API with Supabase integration, health checks, and prediction endpoints

### Web Applications

- **`app.py`**: Streamlit dashboard for interactive housing price predictions
  - Real-time predictions via FastAPI integration
  - Interactive filtering by year, month, and region
  - Visualization of predictions vs actuals with metrics (MAE, RMSE, % Error)
  - Yearly trend analysis with highlighted selected periods
  - **Optimized for Cloud Run**: Uses lazy loading and caching for performance

### Cloud Infrastructure & Deployment

**Migrated from AWS to Google Cloud Run + Supabase Stack**

- **Supabase**:
  - **Storage**: Replaces S3 for storing models (`xgb_best_model.pkl`), encoders, and datasets.
  - **Database**: Stores real-time prediction logs for monitoring.
- **Google Cloud Run**: Serverless container hosting for both the API and Dashboard.
  - **Scale-to-Zero**: auto-scaling to save costs.
  - **Continuous Deployment**: Automated builds connected directly to GitHub.
- **Docker**: Optimized multi-stage builds for fast startup times.

#### Cloud Run Services:

- **housing-api**: FastAPI backend (Autoscaling, Port 8000)
- **housing-streamlit**: Streamlit dashboard (Autoscaling, Port 8501, 2GB Memory)

### Data Leakage Prevention

The project implements strict data leakage prevention:

- Time-based splits (not random)
- Encoders fitted only on training data
- Leakage-prone columns dropped before training
- Schema alignment enforced between train/eval/inference

## Common Commands

### Environment Setup

```bash
# Install dependencies using uv
uv sync
```

### Testing

```bash
# Run all tests
pytest

# Run specific test modules
pytest tests/test_features.py
pytest tests/test_training.py
pytest tests/test_inference.py

# Run with verbose output
pytest -v
```

### Data Pipeline

```bash
# 1. Load and split raw data
python src/feature_pipeline/load.py

# 2. Preprocess splits
python -m src.feature_pipeline.preprocess

# 3. Feature engineering
python -m src.feature_pipeline.feature_engineering
```

### Training Pipeline

```bash
# Train baseline model
python src/training_pipeline/train.py

# Hyperparameter tuning with MLflow
python src/training_pipeline/tune.py

# Model evaluation
python src/training_pipeline/eval.py
```

### Inference

```bash
# Single inference
python src/inference_pipeline/inference.py --input data/raw/holdout.csv --output predictions.csv
```

### API Service

```bash
# Start FastAPI server locally
uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Streamlit Dashboard

```bash
# Start Streamlit dashboard locally
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Docker

```bash
# Build API container
docker build -t housing-regression .

# Build Streamlit container
docker build -t housing-streamlit -f Dockerfile.streamlit .

# Run API container
docker run -p 8000:8000 housing-regression

# Run Streamlit container
docker run -p 8501:8501 housing-streamlit
```

### MLflow Tracking

```bash
# Start MLflow UI (view experiments)
mlflow ui
```

## Key Design Patterns

### Pipeline Modularity

Each pipeline component can be run independently with consistent interfaces. All modules accept configurable input/output paths for testing isolation.

### Cloud-Native Architecture

- **Supabase-First Storage**: Models and data automatically sync from Supabase buckets
- **Containerized Services**: Both API and dashboard run in Docker containers
- **Auto-scaling Infrastructure**: Google Cloud Run provides serverless container scaling
- **Environment-based Configuration**: Separate configs for local development and production

### Encoder Persistence

Frequency and target encoders are saved as pickle files during training and loaded during inference to ensure consistent transformations.

### Configuration Management

Model parameters, file paths, and pipeline settings use sensible defaults but can be overridden through function parameters or environment variables. Production deployments use Cloud Run environment variables.

### Testing Strategy

- Unit tests for individual pipeline components
- Integration tests for end-to-end pipeline flows
- Smoke tests for inference pipeline
- All tests use temporary directories to avoid touching production data

## Dependencies

Key production dependencies (see `pyproject.toml`):

- **ML/Data**: `xgboost==3.0.4`, `scikit-learn`, `pandas==2.1.1`, `numpy==1.26.4`
- **API**: `fastapi`, `uvicorn`
- **Dashboard**: `streamlit`, `plotly`
- **Cloud**: `supabase` (Storage & DB)
- **Experimentation**: `mlflow`, `optuna`
- **Quality**: `great-expectations`, `evidently`

## File Structure Notes

- **`data/`**: Raw, processed, and prediction data (time-structured, Supabase-synced)
- **`models/`**: Trained models and encoders (pkl files, Supabase-synced)
- **`src/utils/supabase_client.py`**: Centralized authentication logic
- **`mlruns/`**: MLflow experiment tracking data
- **`configs/`**: YAML configuration files
- **`notebooks/`**: Jupyter notebooks for EDA and experimentation
- **`08_Supabase_push_dataset.ipynb`**: Migration script for uploading assets
- **`tests/`**: Comprehensive test suite with sample data

## Upcoming Features (Planned)

### 1. API Security

- Implement API Key authentication to protect the endpoint from unauthorized access.
- Store keys securely in Cloud Run Secrets (`API_AUTH_TOKEN`).

### 2. Robust Model Loading ("Bake-in")

- Optimize Docker builds to download model artifacts and encoders during the build phase.
- Improves startup time and eliminates "Cold Start" download latency.

### 3. Drift Detection & Monitoring

- Integrate **Evidently** to compare incoming production data against training baselines.
- Schedule weekly drift reports to detect changes in housing market patterns.
- Alerting system for model degradation.

### 4. Input Validation (Pydantic)

- Implement strict Pydantic schemas for the FastAPI input.
- Provide descriptive `422 Validation Error` responses for missing or incorrect fields (e.g., missing `city` or invalid `zipcode`).
- Catch errors at the API layer before they reach the inference pipeline.
