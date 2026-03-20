"""
Glass Expert AI — File Upload Router
Upload PDF/DOCX/TXT/CSV files, ingest into knowledge base, query immediately.
"""
import os
import uuid
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from loguru import logger
from api.auth import get_current_user, UserInToken

router = APIRouter()

# Temp upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".csv", ".json"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post("/upload")
async def upload_and_ingest(
    file: UploadFile = File(...),
    source_type: str = Form("paper"),
    user: UserInToken = Depends(get_current_user),
):
    """
    Upload a document, ingest it into the knowledge base, and return chunk count.
    Supported: PDF, DOCX, TXT, CSV, JSON (max 50MB).
    """
    # Validate extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Validate source_type
    valid_types = {"textbook", "paper", "sop", "qa_pair", "manual", "standard"}
    if source_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid source_type. Must be one of: {', '.join(valid_types)}",
        )

    # Save to temp file
    file_id = uuid.uuid4().hex[:12]
    safe_name = f"{file_id}_{file.filename}"
    file_path = UPLOAD_DIR / safe_name

    try:
        # Read and check size
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB.",
            )

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"Upload: {file.filename} ({len(content):,} bytes) by user {user.email}")

        # Run ingestion pipeline
        from ingestion.ingest import ingest_document
        chunks_stored = ingest_document(str(file_path), source_type)

        return {
            "status": "success",
            "file_name": file.filename,
            "source_type": source_type,
            "chunks_stored": chunks_stored,
            "message": f"Ingested {chunks_stored} chunks from '{file.filename}'",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
    finally:
        # Clean up temp file
        if file_path.exists():
            file_path.unlink()
