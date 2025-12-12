# ---------------------------------------------------------
# MULTI-STAGE BUILD: Frontend + Backend + Caddy (Railway)
# ---------------------------------------------------------

############ FRONTEND BUILD ############
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

############ BACKEND BUILD ############
FROM python:3.12-slim AS backend-builder
WORKDIR /app

# Build dependencies for numpy/psycopg2/cryptography etc.
RUN apt-get update && apt-get install -y \
    build-essential gcc g++ curl libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

############ FINAL RUNTIME IMAGE ############
FROM python:3.12-slim AS runtime
WORKDIR /app

# Install runtime dependencies (libpq for psycopg2, curl for healthcheck)
RUN apt-get update && apt-get install -y \
    curl libpq5 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Caddy (static binary)
RUN curl -L "https://caddyserver.com/api/download?os=linux&arch=amd64" \
        -o /tmp/caddy.tar.gz && \
    tar -xzf /tmp/caddy.tar.gz -C /usr/bin caddy && \
    chmod +x /usr/bin/caddy && \
    rm -rf /tmp/caddy.tar.gz

# Copy Python packages from builder
COPY --from=backend-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy backend code
COPY --from=backend-builder /app /app

# Copy built frontend
COPY --from=frontend-builder /frontend/dist /srv

# Copy Caddyfile
COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE 80

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Create startup script
RUN printf '#!/bin/sh\n\
set -e\n\
export PYTHONUNBUFFERED=1\n\
echo "Starting FastAPI backend..."\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 &\n\
BACKEND_PID=$!\n\
sleep 2\n\
echo "Starting Caddy..."\n\
caddy run --config /etc/caddy/Caddyfile &\n\
CADDY_PID=$!\n\
wait $BACKEND_PID $CADDY_PID\n' > /start.sh \
    && chmod +x /start.sh

CMD ["/start.sh"]