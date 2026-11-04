# Multi-stage Dockerfile: Builds React frontend and serves via FastAPI

# --- Stage 1: Build React Frontend ---
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python Runtime Environment ---
FROM python:3.12-slim AS runner
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Download required NLTK data for financial sentiment
RUN python -c "import nltk; nltk.download('vader_lexicon', quiet=True); nltk.download('punkt', quiet=True)"

# Ensure runtime data and model directories exist
RUN mkdir -p data output/data output/models

# Copy backend code, models, data, and scripts
COPY backend/ ./backend/
COPY src/ ./src/
COPY data/ ./data/
COPY output/ ./output/
COPY scripts/ ./scripts/

# Copy built frontend assets from Stage 1 into frontend/dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port (default 8000, overridable by $PORT on Render/Hugging Face)
ENV PORT=8000
EXPOSE 8000

# Start Uvicorn server binding to 0.0.0.0 and dynamic port (defaults to 8000)
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

