# ---------------------------------------------------
# Inference / Runtime Dockerfile
# ---------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# curl/git might be needed for some pip installs or healthchecks
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 1. Install Base Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Install PyTorch (CPU only) to save massive space
# We install this BEFORE requirements-ml.txt so sentence-transformers sees it
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install ML Dependencies (Inference)
COPY requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy application
COPY . .

# Ensure data directories exist (they might be mounted)
RUN mkdir -p data models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]