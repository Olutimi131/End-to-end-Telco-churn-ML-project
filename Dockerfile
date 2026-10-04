# ==============================================================================
# Production Dockerfile for Telco Customer Churn Intelligence
# Supports FastAPI Backend (port 8000) and Streamlit Frontend (port 8501)
# ==============================================================================
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PORT=8000 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

WORKDIR /app

# Install operating system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Run data ingestion and training if model artifacts do not exist
RUN python -c "from pathlib import Path; import subprocess; \
    subprocess.run(['python', 'src/data_ingestion.py']) if not Path('data/processed/train.csv').exists() else None; \
    subprocess.run(['python', 'src/train.py']) if not Path('models/churn_pipeline.joblib').exists() else None"

# Expose FastAPI backend and Streamlit frontend ports
EXPOSE 8000 8501

# Health check against FastAPI backend
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default entry command: Launch production FastAPI backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
