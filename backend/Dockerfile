FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NOTE: ML/training dependencies (sentence-transformers, torch, faiss, etc.)
# have been moved to `requirements-ml.txt` to keep runtime image small.
# If you need to run training inside a container, use `Dockerfile.train`.

# Copy application
COPY . .

# Create data directory for FAISS index
RUN mkdir -p data

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
