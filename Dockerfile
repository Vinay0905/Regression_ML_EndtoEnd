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

# Cloud Run injects a PORT environment variable. We set a default for local testing.
ENV PORT=8000
EXPOSE 8000

# We use shell form to allow environment variable expansion for $PORT
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]