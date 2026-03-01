"""
Glass Expert AI — FastAPI Backend
Phase 2: RAG Query API
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables from .env BEFORE importing routers
load_dotenv(Path(__file__).parent.parent / ".env")

from api.routers import query, health, ingest

app = FastAPI(
    title="Glass Expert AI",
    description="RAG-powered glass science assistant for engineers",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(query.router,  prefix="/api/v1", tags=["Query"])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingestion"])


@app.on_event("startup")
async def startup_event():
    logger.info("Glass Expert AI API starting up...")
    logger.info("Docs available at: http://localhost:8080/docs")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8080)),
        reload=True,
        log_level="info",
    )
