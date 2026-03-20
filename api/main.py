"""
Glass Expert AI — FastAPI Backend
Phase 3: RAG Query API + Specialized Engineering Endpoints + React Chat UI
"""
import os
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from dotenv import load_dotenv
from starlette.middleware.gzip import GZipMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables BEFORE importing routers
load_dotenv(Path(__file__).parent.parent / ".env")

from api.routers import query, health, ingest, analyze, design, troubleshoot, feedback, conversations, upload, export, admin
from api.auth import router as auth_router

app = FastAPI(
    title="Glass Expert AI",
    debug=True,
    description=(
        "RAG-powered glass science assistant for engineers. "
        "Endpoints: /query (general Q&A), /analyze (composition analysis), "
        "/design (composition design), /troubleshoot (defect root cause)."
    ),
    version="3.1.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
_allowed_origins = os.getenv("CORS_ORIGINS", "").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]
if not _allowed_origins:
    _allowed_origins = [
        "http://localhost:8080",
        "http://localhost:3000",
        "http://127.0.0.1:8080",
    ]

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Rate limiting ──────────────────────────────────────────────────────────────
_rate_limits: dict = defaultdict(list)
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    _rate_limits[client_ip] = [
        t for t in _rate_limits[client_ip] if now - t < _RATE_LIMIT_WINDOW
    ]
    if len(_rate_limits[client_ip]) >= _RATE_LIMIT_MAX:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please wait before trying again."},
            headers={"Retry-After": str(_RATE_LIMIT_WINDOW)},
        )
    _rate_limits[client_ip].append(now)
    return await call_next(request)


# ── Request ID + process time headers ─────────────────────────────────────────
@app.middleware("http")
async def add_request_metadata(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
        process_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = f"{process_ms:.0f}ms"
        return response
    except Exception as e:
        logger.error(f"Middleware error: {e}")
        process_ms = (time.perf_counter() - start) * 1000
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={"X-Request-ID": request_id, "X-Process-Time": f"{process_ms:.0f}ms"},
        )


# ── Auth ───────────────────────────────────────────────────────────────────────
app.include_router(auth_router,          prefix="/api/v1", tags=["Auth"])

# ── Core endpoints ─────────────────────────────────────────────────────────────
app.include_router(health.router,        prefix="/api/v1", tags=["Health"])
app.include_router(query.router,         prefix="/api/v1", tags=["Query"])
app.include_router(ingest.router,        prefix="/api/v1", tags=["Ingestion"])

# ── Specialized engineering endpoints ─────────────────────────────────────────
app.include_router(analyze.router,       prefix="/api/v1", tags=["Analyze"])
app.include_router(design.router,        prefix="/api/v1", tags=["Design"])
app.include_router(troubleshoot.router,  prefix="/api/v1", tags=["Troubleshoot"])
app.include_router(feedback.router,      prefix="/api/v1", tags=["Feedback"])
app.include_router(conversations.router, prefix="/api/v1", tags=["Conversations"])
app.include_router(upload.router,        prefix="/api/v1", tags=["Upload"])
app.include_router(export.router,        prefix="/api/v1", tags=["Export"])
app.include_router(admin.router,         prefix="/api/v1", tags=["Admin"])

# ── Serve React frontend ───────────────────────────────────────────────────────
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(_frontend_dist / "assets")),
        name="assets",
    )

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        return FileResponse(str(_frontend_dist / "favicon.svg"))

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(_frontend_dist / "index.html"))


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    from api.database import _init_pool
    try:
        _init_pool()
    except Exception as e:
        logger.error(f"Failed to initialize DB pool on startup: {e}")

    logger.info("Glass Expert AI API v3.1.0 starting up...")
    logger.info(f"CORS origins: {_allowed_origins}")
    logger.info(f"Rate limit: {_RATE_LIMIT_MAX} req/min")
    logger.info("Chat UI available at: http://localhost:8080/")
    logger.info("API Docs available at: http://localhost:8080/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8080)),
        reload=True,
        log_level="info",
    )