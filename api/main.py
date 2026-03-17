
"""
Glass Expert AI — FastAPI Backend
Phase 2: RAG Query API + Specialized Engineering Endpoints
"""
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv(Path(__file__).parent.parent / ".env")

from api.routers import (
    query, health, ingest, analyze, design,
    troubleshoot, feedback, conversations
)

from api.auth import router as auth_router

app = FastAPI(
    title="Glass Expert AI",
    description=(
        "RAG-powered glass science assistant for engineers. "
        "Endpoints: /query /analyze /design /troubleshoot /feedback /auth /conversations"
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth ───────────────────────────────────────────────────────────────────────
app.include_router(auth_router,             prefix="/api/v1", tags=["Auth"])
# ── Core ───────────────────────────────────────────────────────────────────────
app.include_router(health.router,           prefix="/api/v1",      tags=["Health"])
app.include_router(query.router,            prefix="/api/v1",      tags=["Query"])
app.include_router(ingest.router,           prefix="/api/v1",      tags=["Ingestion"])

# ── Specialized engineering ────────────────────────────────────────────────────
app.include_router(analyze.router,          prefix="/api/v1",      tags=["Analyze"])
app.include_router(design.router,           prefix="/api/v1",      tags=["Design"])
app.include_router(troubleshoot.router,     prefix="/api/v1",      tags=["Troubleshoot"])
app.include_router(feedback.router,         prefix="/api/v1",      tags=["Feedback"])

# ── Conversations & Memory ─────────────────────────────────────────────────────
app.include_router(conversations.router,    prefix="/api/v1",      tags=["Conversations"])

# ── Serve React frontend ───────────────────────────────────────────────────────
_project_root  = Path(__file__).resolve().parent.parent
_frontend_dist = _project_root / "frontend" / "dist"
_assets_dir    = _frontend_dist / "assets"
_index_html    = _frontend_dist / "index.html"


@app.get("/", include_in_schema=False)
async def serve_frontend():
    if _index_html.exists():
        return FileResponse(str(_index_html))
    return HTMLResponse(
        "<h1>Glass Expert AI</h1>"
        "<p>Frontend not built. Run: cd frontend && npm run build</p>"
    )


@app.get("/favicon.svg", include_in_schema=False)
async def favicon():
    f = _frontend_dist / "favicon.svg"
    if f.exists():
        return FileResponse(str(f))
    return FileResponse(str(_index_html))


if _assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")


@app.on_event("startup")
async def startup_event():
    logger.info("Glass Expert AI API v3.0.0 starting up...")
    logger.info("API Docs: http://localhost:8080/docs")
    logger.info(f"Frontend: {_frontend_dist} (exists={_frontend_dist.exists()})")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8080)),
        reload=True,
        log_level="info",
    )