# ============================================================
# Glass Expert AI — API Server Dockerfile
# Multi-stage: build React frontend, then run FastAPI backend
# ============================================================

# ── Stage 1: Build React frontend ────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python API server ───────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2-binary and general build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY retrieval/ ./retrieval/
COPY ingestion/ ./ingestion/
COPY scripts/ ./scripts/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Env defaults (override at runtime)
ENV API_HOST=0.0.0.0
ENV API_PORT=8080
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
