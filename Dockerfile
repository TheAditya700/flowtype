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
    build-essential gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

############ FINAL RUNTIME IMAGE ############
FROM python:3.12-slim AS runtime
WORKDIR /app

# Install Caddy (static binary — no apt repo needed)
RUN apt-get update && apt-get install -y curl && \
    curl -L "https://caddyserver.com/api/download?os=linux&arch=amd64" \
        -o /tmp/caddy.tar.gz && \
    tar -xzf /tmp/caddy.tar.gz -C /usr/bin caddy && \
    chmod +x /usr/bin/caddy && \
    rm -rf /tmp/caddy.tar.gz && \
    apt-get purge -y curl && apt-get autoremove -y

# Copy backend dependencies and code
COPY --from=backend-builder /usr/local /usr/local
COPY --from=backend-builder /app /app

# Copy built frontend
COPY --from=frontend-builder /frontend/dist /srv

# Copy Caddyfile
COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE 80

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Start both backend + Caddy
RUN echo '#!/bin/sh\n\
export PYTHONUNBUFFERED=1\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 &\n\
exec caddy run --config /etc/caddy/Caddyfile\n' > /start.sh \
    && chmod +x /start.sh

CMD ["/start.sh"]
