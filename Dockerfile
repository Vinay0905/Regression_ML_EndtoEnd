# Use slim Python base image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependency files first
COPY pyproject.toml uv.lock* ./

# Install uv and dependencies
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy project files
COPY . .

# Cloud Run injects the PORT environment variable.
ENV PORT=8000
EXPOSE 8000

# Use shell form to allow $PORT expansion.
CMD .venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT