# ---------------------------------------------------------
# MULTI-STAGE BUILD: Frontend + Backend + Caddy (Railway)
# ---------------------------------------------------------

############################
# FRONTEND BUILD
############################
FROM node:20-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install --production=false
COPY frontend/ .
RUN npm run build

############################
# BACKEND BUILD
############################
FROM python:3.12-slim AS backend-builder
WORKDIR /app

# Install system deps for psycopg2, faiss, numpy, cryptography
RUN apt-get update && apt-get install -y \
    gcc g++ curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

############################
# FINAL RUNTIME IMAGE
############################
FROM caddy:2-alpine

# Install Python runtime + dumb-init
RUN apk add --no-cache python3 py3-pip dumb-init

# Install minimal Python dependencies into runtime image
COPY --from=backend-builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy Caddyfile
COPY Caddyfile /etc/caddy/Caddyfile

# Copy built frontend
COPY --from=frontend-builder /frontend/dist /srv

# Copy backend code
COPY --from=backend-builder /app /app

WORKDIR /app

EXPOSE 80 443

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# Combined startup script
RUN echo '#!/bin/sh\n\
uvicorn app.main:app --host 127.0.0.1 --port 8000 &\n\
exec caddy run --config /etc/caddy/Caddyfile\n' \
> /start.sh && chmod +x /start.sh

ENTRYPOINT ["/sbin/dumb-init", "--"]
CMD ["/start.sh"]
