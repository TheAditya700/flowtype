# ---------------------------------------------------
# Multi-stage build for frontend production bundle (v0.3.0 MVP)
# ---------------------------------------------------
FROM node:20-alpine AS builder

ARG VITE_API_URL=http://localhost:8000/api
ENV VITE_API_URL=${VITE_API_URL}

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --prefer-offline --no-audit

# Copy source
COPY . .

# Build for production (VITE_API_URL injected via ENV)
RUN npm run build

# Production image: serve static files
FROM node:20-alpine

WORKDIR /app

# Install a simple static server
RUN npm install -g serve

# Copy built app from builder
COPY --from=builder /app/dist ./dist

EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost/ || exit 1

# Serve static files
CMD ["serve", "-s", "dist", "-l", "80"]