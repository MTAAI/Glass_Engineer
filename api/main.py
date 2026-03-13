"""
Glass Expert AI — FastAPI Backend
Phase 3: RAG Query API + Specialized Engineering Endpoints + React Chat UI
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from loguru import logger
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env BEFORE importing routers
load_dotenv(Path(__file__).parent.parent / ".env")

from api.routers import query, health, ingest, analyze, design, troubleshoot, feedback, conversations
from api.auth import router as auth_router

app = FastAPI(
    title="Glass Expert AI",
    description=(
        "RAG-powered glass science assistant for engineers. "
        "Endpoints: /query (general Q&A), /analyze (composition analysis), "
        "/design (composition design), /troubleshoot (defect root cause)."
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
app.include_router(auth_router,         prefix="/api/v1", tags=["Auth"])

# ── Core endpoints ─────────────────────────────────────────────────────────────
app.include_router(health.router,       prefix="/api/v1", tags=["Health"])
app.include_router(query.router,        prefix="/api/v1", tags=["Query"])
app.include_router(ingest.router,       prefix="/api/v1", tags=["Ingestion"])

# ── Specialized engineering endpoints ─────────────────────────────────────────
app.include_router(analyze.router,      prefix="/api/v1", tags=["Analyze"])
app.include_router(design.router,       prefix="/api/v1", tags=["Design"])
app.include_router(troubleshoot.router, prefix="/api/v1", tags=["Troubleshoot"])
app.include_router(feedback.router,     prefix="/api/v1", tags=["Feedback"])
app.include_router(conversations.router, prefix="/api/v1", tags=["Conversations"])

# ── Serve React frontend ───────────────────────────────────────────────────────
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    @app.get("/favicon.svg", include_in_schema=False)
    async def favicon():
        return FileResponse(str(_frontend_dist / "favicon.svg"))

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(_frontend_dist / "index.html"))


@app.on_event("startup")
async def startup_event():
    logger.info("Glass Expert AI API v3.0.0 starting up...")
    logger.info("Chat UI available at: http://localhost:8080/")
    logger.info("API Docs available at: http://localhost:8080/docs")
    logger.info("Endpoints: /analyze, /design, /troubleshoot, /feedback")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8080)),
        reload=True,
        log_level="info",
    )