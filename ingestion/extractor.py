"""
Glass Expert AI — Document Extractor v2
Extracts clean text from: PDF, CSV, NPZ, TXT, DOCX, JSON
Supports English and Farsi (RTL) with automatic language detection.
"""
import json
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from pathlib import Path
from langdetect import detect, LangDetectException
from loguru import logger

# ── Supported file types ───────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".npz", ".txt", ".docx", ".json"}


def extract_file(file_path: str) -> dict:
    """
    Auto-detect file type and extract text accordingly.

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        dict with keys: title, file_path, language, text, page_count, char_count
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    logger.info(f"Extracting [{ext.upper()}]: {path.name}")

    if ext == ".pdf":
        return extract_pdf(file_path)
    elif ext == ".csv":
        return extract_csv(file_path)
    elif ext == ".npz":
        return extract_npz(file_path)
    elif ext == ".txt":
        return extract_txt(file_path)
    elif ext == ".docx":
        return extract_docx(file_path)
    elif ext == ".json":
        return extract_json(file_path)


# ── PDF Extractor ──────────────────────────────────────────────────────────────
def extract_pdf(file_path: str) -> dict:
    """Extract text from a PDF file with automatic language detection."""
    path = Path(file_path)
    doc = fitz.open(str(path))
    pages_text = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        text = _clean_page_text(text, page_num + 1)
        if text.strip():
            pages_text.append(text)

    doc.close()
    full_text = "\n\n".join(pages_text)
    language = _detect_language(full_text[:1000])

    result = {
        "title": path.stem,
        "file_path": str(path.resolve()),
        "language": language,
        "text": full_text,
        "page_count": len(pages_text),
        "char_count": len(full_text),
    }
    logger.info(
        f"Extracted {result['char_count']:,} characters from {result['page_count']} pages "
        f"| Language: {language.upper()}"
    )
    return result


# ── CSV Extractor ──────────────────────────────────────────────────────────────
def extract_csv(file_path: str) -> dict:
    """
    Extract text from a CSV file.
    Converts each row into a natural-language sentence for embedding.
    Ideal for glass composition databases, property tables, and QA pairs.
    """
    path = Path(file_path)
    df = pd.read_csv(str(path), encoding="utf-8", on_bad_lines="skip")

    lines = []
    lines.append(f"Dataset: {path.stem}")
    lines.append(f"Columns: {', '.join(df.columns.tolist())}")
    lines.append(f"Total records: {len(df)}")
    lines.append("")

    for idx, row in df.iterrows():
        parts = []
        for col, val in row.items():
            if pd.notna(val) and str(val).strip():
                parts.append(f"{col}: {val}")
        if parts:
            lines.append(" | ".join(parts))

    full_text = "\n".join(lines)
    language = _detect_language(full_text[:500])

    result = {
        "title": path.stem,
        "file_path": str(path.resolve()),
        "language": language,
        "text": full_text,
        "page_count": 1,
        "char_count": len(full_text),
        "row_count": len(df),
        "columns": df.columns.tolist(),
    }
    logger.info(
        f"Extracted {len(df)} rows x {len(df.columns)} columns "
        f"= {result['char_count']:,} characters | Language: {language.upper()}"
    )
    return result


# ── NPZ Extractor ──────────────────────────────────────────────────────────────
def extract_npz(file_path: str) -> dict:
    """
    Extract metadata and statistical summaries from NumPy .npz files.
    These typically contain glass property arrays, spectral data, or simulation results.
    Converts array statistics into searchable text rather than raw numbers.
    """
    path = Path(file_path)
    data = np.load(str(path), allow_pickle=True)

    lines = []
    lines.append(f"NumPy Dataset: {path.stem}")
    lines.append(f"Arrays contained: {', '.join(data.files)}")
    lines.append("")

    for array_name in data.files:
        arr = data[array_name]
        lines.append(f"Array: {array_name}")
        lines.append(f"  Shape: {arr.shape}")
        lines.append(f"  Data type: {arr.dtype}")

        if arr.dtype.kind in ("f", "i", "u") and arr.size > 0:
            flat = arr.flatten()
            lines.append(f"  Min: {float(np.min(flat)):.6g}")
            lines.append(f"  Max: {float(np.max(flat)):.6g}")
            lines.append(f"  Mean: {float(np.mean(flat)):.6g}")
            lines.append(f"  Std: {float(np.std(flat)):.6g}")
            sample = flat[:5].tolist()
            lines.append(f"  Sample values: {sample}")
        elif arr.dtype.kind in ("U", "S", "O"):
            try:
                unique_vals = list(set(arr.flatten().tolist()))[:10]
                lines.append(f"  Unique values (sample): {unique_vals}")
            except Exception:
                pass
        lines.append("")

    array_files = list(data.files)
    data.close()
    full_text = "\n".join(lines)

    result = {
        "title": path.stem,
        "file_path": str(path.resolve()),
        "language": "en",
        "text": full_text,
        "page_count": 1,
        "char_count": len(full_text),
        "arrays": array_files,
    }
    logger.info(
        f"Extracted {len(array_files)} arrays = {result['char_count']:,} characters"
    )
    return result


# ── TXT Extractor ──────────────────────────────────────────────────────────────
def extract_txt(file_path: str) -> dict:
    """Extract text from a plain text file."""
    path = Path(file_path)
    text = path.read_text(encoding="utf-8", errors="ignore")
    language = _detect_language(text[:1000])

    result = {
        "title": path.stem,
        "file_path": str(path.resolve()),
        "language": language,
        "text": text,
        "page_count": 1,
        "char_count": len(text),
    }
    logger.info(
        f"Extracted {result['char_count']:,} characters | Language: {language.upper()}"
    )
    return result


# ── DOCX Extractor ─────────────────────────────────────────────────────────────
def extract_docx(file_path: str) -> dict:
    """Extract text from a Microsoft Word .docx file."""
    from docx import Document

    path = Path(file_path)
    doc = Document(str(path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(
                cell.text.strip() for cell in row.cells if cell.text.strip()
            )
            if row_text:
                paragraphs.append(row_text)

    full_text = "\n".join(paragraphs)
    language = _detect_language(full_text[:1000])

    result = {
        "title": path.stem,
        "file_path": str(path.resolve()),
        "language": language,
        "text": full_text,
        "page_count": 1,
        "char_count": len(full_text),
    }
    logger.info(
        f"Extracted {result['char_count']:,} characters | Language: {language.upper()}"
    )
    return result


# ── JSON Extractor ─────────────────────────────────────────────────────────────
def extract_json(file_path: str) -> dict:
    """
    Extract text from a JSON file.
    Handles QA pairs, property databases, and structured glass data.
    """
    path = Path(file_path)
    with open(str(path), encoding="utf-8") as f:
        data = json.load(f)

    lines = [f"JSON Dataset: {path.stem}", ""]

    def flatten_json(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flatten_json(v, f"{prefix}{k}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj[:500]):
                flatten_json(item, f"{prefix}[{i}]")
        else:
            if str(obj).strip():
                lines.append(f"{prefix}: {obj}")

    flatten_json(data)
    full_text = "\n".join(lines)
    language = _detect_language(full_text[:1000])

    result = {
        "title": path.stem,
        "file_path": str(path.resolve()),
        "language": language,
        "text": full_text,
        "page_count": 1,
        "char_count": len(full_text),
    }
    logger.info(
        f"Extracted {result['char_count']:,} characters | Language: {language.upper()}"
    )
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────
def _clean_page_text(text: str, page_num: int) -> str:
    """Remove common PDF artifacts like page numbers and headers/footers."""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.isdigit():
            continue
        if stripped in (f"- {page_num} -", f"— {page_num} —", str(page_num)):
            continue
        if len(stripped) < 4:
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


def _detect_language(text: str) -> str:
    """Detect language of text, defaulting to English if detection fails."""
    try:
        lang = detect(text)
        if lang in ("fa", "ar"):
            return "fa"
        return "en"
    except LangDetectException:
        logger.warning("Language detection failed, defaulting to English.")
        return "en"
