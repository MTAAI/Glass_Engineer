# ============================================================
# Glass Expert AI (Llama 8B) — API Server Dockerfile
# Uses existing docker-api image as base (has torch + deps)
# Only rebuilds: frontend + application code
# ============================================================

# ── Stage 1: Build React frontend ────────────────────────────
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Update code on existing base ─────────────────────
FROM docker-api:latest AS runtime

USER root
WORKDIR /app

# Update any new dependencies (fast — most already installed)
COPY requirements-docker.txt ./requirements-docker.txt
RUN pip install --no-cache-dir --timeout 300 -r requirements-docker.txt 2>/dev/null || true

# Copy updated application code
COPY api/        ./api/
COPY retrieval/  ./retrieval/
COPY ingestion/  ./ingestion/
COPY scripts/    ./scripts/

# Copy fresh frontend build
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Ensure non-root user exists and owns files
RUN (groupadd --gid 1000 appuser 2>/dev/null || true) \
    && (useradd --uid 1000 --gid 1000 --create-home appuser 2>/dev/null || true) \
    && mkdir -p /app/logs && chown -R appuser:appuser /app

USER appuser

ENV API_HOST=0.0.0.0 \
    API_PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/api/v1/health || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "2", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--access-log", \
     "--proxy-headers", \
     "--forwarded-allow-ips", "*"]
