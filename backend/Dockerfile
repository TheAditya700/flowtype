# ---------------------------------------------------
# Stage 1: Build environment (only used for building wheels)
# ---------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system-level build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies
COPY requirements.txt .

# Create venv to isolate packages
RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps into venv (wheels built here)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# ---------------------------------------------------
# Stage 2: Lightweight runtime image
# ---------------------------------------------------
FROM python:3.11-slim

WORKDIR /app

# Install ONLY runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder stage
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

# Copy app code
COPY . .

# Ensure FAISS/models directory exists
RUN mkdir -p /app/data

EXPOSE 8000

# HEALTHCHECK is good
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production Uvicorn (no reload, more efficient)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
