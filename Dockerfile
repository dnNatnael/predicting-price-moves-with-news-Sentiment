# Multi-stage Dockerfile: Builds React frontend and serves via FastAPI
# Optimized for cloud deployments (Render, Hugging Face Spaces, Fly.io, Railway)

# --- Stage 1: Build React Frontend ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python Production Runtime Environment ---
FROM python:3.12-slim AS runner
WORKDIR /app

# Ensure Python outputs are printed unbuffered to console logs
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=10000

# Install minimal system dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install production Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK sentiment dictionaries & tokenizers
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Ensure runtime directories exist
RUN mkdir -p data output/data output/models

# Copy application backend, models, reference data, and source code
COPY backend/ ./backend/
COPY src/ ./src/
COPY data/ ./data/
COPY output/ ./output/
COPY scripts/ ./scripts/

# Copy built static frontend assets from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose Render default web service port
EXPOSE 10000

# Health check to ensure service readiness
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT:-10000}/api/health || exit 1

# Launch Uvicorn using exec so signals and stdout stream properly
CMD ["sh", "-c", "exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
