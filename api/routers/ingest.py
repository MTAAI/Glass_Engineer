"""
Glass Expert AI — Ingestion Router
Trigger document ingestion via API.
"""
from pathlib import Path
from fastapi import APIRouter, HTTPException
from loguru import logger
from api.models.schemas import IngestRequest, IngestResponse

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest a single document into the knowledge base.
    Supports: PDF, CSV, NPZ, TXT, DOCX, JSON
    """
    from ingestion.ingest import ingest_document as run_ingest

    file_path = Path(request.file_path)

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {request.file_path}",
        )

    valid_types = {"textbook", "paper", "sop", "qa_pair", "manual", "standard"}
    if request.source_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_type. Must be one of: {', '.join(valid_types)}",
        )

    try:
        result = run_ingest(str(file_path), request.source_type)
        return IngestResponse(
            file_name=file_path.name,
            source_type=request.source_type,
            chunks_stored=result.get("chunks", 0),
            language=result.get("language", "en"),
            status="success",
            message=f"Successfully ingested {result.get('chunks', 0)} chunks",
        )
    except Exception as e:
        logger.error(f"Ingestion API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
