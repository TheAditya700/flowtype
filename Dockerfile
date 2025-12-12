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

RUN apk add --no-cache python3 py3-pip dumb-init

COPY Caddyfile /etc/caddy/Caddyfile

COPY --from=frontend-builder /frontend/dist /srv
COPY --from=backend-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=backend-builder /app /app

WORKDIR /app

EXPOSE 80 443

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

RUN echo '#!/bin/sh\n\
cd /app\n\
export PYTHONUNBUFFERED=1\n\
uvicorn app.main:app --host 127.0.0.1 --port 8000 &\n\
exec caddy run --config /etc/caddy/Caddyfile\n\
' > /start.sh && chmod +x /start.sh

ENTRYPOINT ["dumb-init", "--"]
CMD ["/start.sh"]
