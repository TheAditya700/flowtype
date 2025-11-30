# Multi-stage build for frontend production bundle
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy source
COPY . .

# Build for production
RUN npm run build

# Production image: serve static files
FROM node:20-alpine

WORKDIR /app

# Install a simple static server
RUN npm install -g serve

# Copy built app from builder
COPY --from=builder /app/dist ./dist

# Expose port
EXPOSE 3000

# Serve static files
CMD ["serve", "-s", "dist", "-l", "3000"]
