# ---------------------------------------------------------
# MULTI-STAGE BUILD: Frontend + Backend + Caddy (Railway)
# Final image uses Debian Slim for compatibility
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

# System deps needed for psycopg2, numpy, faiss, cryptography
RUN apt-get update && apt-get install -y \
    gcc g++ curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

############################
# FINAL RUNTIME IMAGE
############################
FROM python:3.12-slim AS runtime

# Install Caddy
RUN apt-get update && apt-get install -y debian-keyring debian-archive-keyring curl \
    && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | apt-key add - \
    && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list \
    && apt-get update && apt-get install -y caddy dumb-init \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Copy frontend
COPY --from=frontend-builder /frontend/dist /srv

# Copy backend runtime + libs
COPY --from=backend-builder /usr/local /usr/local
COPY --from=backend-builder /app /app

WORKDIR /app

EXPOSE 80 443

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost/ || exit 1

# Start backend + Caddy
RUN echo '#!/bin/sh\n\
cd /app\n\
export PYTHONUNBUFFERED=1\n\
uvicorn app.main:app --host 127.0.0.1 --port 8000 &\n\
exec caddy run --config /etc/caddy/Caddyfile\n\
' > /start.sh && chmod +x /start.sh

ENTRYPOINT ["dumb-init", "--"]
CMD ["/start.sh"]
