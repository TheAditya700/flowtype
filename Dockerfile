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

RUN apt-get update && apt-get install -y \
    build-essential gcc g++ curl libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

############ FINAL RUNTIME IMAGE ############
FROM python:3.12-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl libpq5 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Caddy binary
RUN curl -L "https://caddyserver.com/api/download?os=linux&arch=amd64" \
        -o /usr/bin/caddy && \
    chmod +x /usr/bin/caddy

# Copy Python packages
COPY --from=backend-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy backend code
COPY --from=backend-builder /app /app

# Copy frontend build output
COPY --from=frontend-builder /frontend/dist /srv

# Copy Caddy config
COPY Caddyfile /etc/caddy/Caddyfile

EXPOSE 80

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# ---------------------------------------------------------
# Start Script: RUN ALEMBIC MIGRATIONS + BACKEND + CADDY
# ---------------------------------------------------------
RUN printf '#!/bin/sh\n\
set -e\n\
export PYTHONUNBUFFERED=1\n\
\n\
echo \"Running Alembic migrations...\"\n\
if [ -d \"/app/backend/migrations\" ]; then\n\
    cd /app/backend\n\
    alembic upgrade head || echo \"Alembic failed, continuing startup...\"\n\
else\n\
    echo \"No migrations directory found, skipping Alembic.\"\n\
fi\n\
\n\
echo \"Starting FastAPI backend...\"\n\
uvicorn app.main:app --host 0.0.0.0 --port 8000 &\n\
BACKEND_PID=$!\n\
sleep 2\n\
\n\
echo \"Starting Caddy...\"\n\
caddy run --config /etc/caddy/Caddyfile &\n\
CADDY_PID=$!\n\
\n\
wait $BACKEND_PID $CADDY_PID\n' > /start.sh \
    && chmod +x /start.sh

CMD ["/start.sh"]
