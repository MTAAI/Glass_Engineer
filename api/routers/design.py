"""
Glass Expert AI — Design Router
POST /api/v1/design
Suggests glass compositions to meet target property requirements.
"""
import time
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from api.auth import get_current_user, UserInToken

from api.models.schemas import DesignRequest, DesignResponse, SourceChunk

router = APIRouter()


def _build_design_query(request: DesignRequest) -> str:
    """Build a retrieval query from the design request."""
    parts = ["glass composition design"]
    for key, value in request.target_properties.items():
        parts.append(f"{key} {value}")
    if request.base_system:
        parts.append(request.base_system)
    if request.constraints:
        parts.extend(request.constraints[:3])
    return " ".join(str(p) for p in parts)


def _build_design_prompt(request: DesignRequest, context: str) -> str:
    """Build the LLM prompt for composition design."""
    props_str = "\n".join(
        f"  - {k}: {v}" for k, v in request.target_properties.items()
    )
    constraints_str = ""
    if request.constraints:
        constraints_str = "\nConstraints:\n" + "\n".join(
            f"  - {c}" for c in request.constraints
        )
    base_str = f"\nBase Glass System: {request.base_system}" if request.base_system else ""

    return f"""You are Glass Expert AI. Design glass compositions to meet the following target properties.

KNOWLEDGE BASE CONTEXT:
{context}

TARGET PROPERTIES:
{props_str}
{base_str}
{constraints_str}

Provide:
1. SUGGESTED COMPOSITIONS: Give 2-3 candidate compositions as oxide weight percentages (SiO2, Na2O, CaO, etc.). For each, explain why it meets the targets.
2. DESIGN RATIONALE: Explain the scientific reasoning behind the composition choices — which oxides control which properties.
3. TRADE-OFFS: List any compromises or trade-offs between the target properties.
4. MANUFACTURING NOTES: Key process considerations for melting, forming, and annealing these compositions.

Be specific with weight percentages and cite knowledge base sources where possible."""


@router.post("/design", response_model=DesignResponse)
async def design_glass(request: DesignRequest, user: UserInToken = Depends(get_current_user)):
    """
    Design a glass composition to meet target properties.

    Provide `target_properties` as a dict of property name → target value.
    Optionally specify `constraints` (e.g. no lead, cost limits) and `base_system`.

    Example target properties:
    - Tg_min_C: 500
    - CTE_max_ppm_K: 9.0
    - visible_transmittance_min_pct: 90
    - application: "architectural float glass"
    """
    from retrieval.retriever import retrieve_with_auto_language, format_context

    if not request.target_properties:
        raise HTTPException(
            status_code=422,
            detail="target_properties must not be empty"
        )

    start_time = time.time()

    # ── Step 1: Build retrieval query ─────────────────────────────────────────
    query = _build_design_query(request)

    # ── Step 2: Retrieve relevant knowledge ───────────────────────────────────
    try:
        chunks, _ = retrieve_with_auto_language(query=query, top_k=8)
    except Exception as e:
        logger.error(f"Design retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    retrieval_time_ms = (time.time() - start_time) * 1000
    context_block = format_context(chunks) if chunks else "No relevant knowledge base entries found."

    # ── Step 3: Generate design with LLM ──────────────────────────────────────
    prompt = _build_design_prompt(request, context_block)

    try:
        from retrieval.llm import generate_answer
        design_text, model_used = await generate_answer(
            question=prompt,
            context=context_block,
            language="en",
        )
    except Exception as e:
        logger.error(f"Design LLM error: {e}")
        design_text = "LLM generation failed. Retrieved context:\n\n" + context_block
        model_used = "retrieval-only"

    # ── Step 4: Parse structured output ───────────────────────────────────────
    suggested_compositions = _parse_compositions(design_text)
    trade_offs = _parse_section(design_text, "TRADE-OFFS", "TRADE_OFFS", "TRADEOFFS")
    manufacturing_notes = _parse_section_text(design_text, "MANUFACTURING NOTES", "MANUFACTURING")

    # ── Step 5: Build response ─────────────────────────────────────────────────
    sources = [
        SourceChunk(
            title=c.get("title", "Unknown"),
            source_type=c.get("source_type", "unknown"),
            language=c.get("language", "en"),
            similarity=round(c.get("similarity", 0.0), 4),
            content_preview=c.get("content", "")[:300] + "...",
        )
        for c in chunks
    ]

    return DesignResponse(
        suggested_compositions=suggested_compositions if suggested_compositions else [{"note": "See full design rationale above"}],
        design_rationale=design_text,
        trade_offs=trade_offs if trade_offs else ["See full analysis for trade-off details"],
        manufacturing_notes=manufacturing_notes if manufacturing_notes else "See full analysis for manufacturing notes",
        sources=sources,
        model_used=model_used,
        retrieval_time_ms=round(retrieval_time_ms, 2),
    )


def _parse_compositions(text: str) -> list:
    """Extract composition dicts from LLM output."""
    import re
    compositions = []
    # Look for patterns like SiO2: 72%, Na2O: 14%
    blocks = re.findall(
        r"(?:Composition|Option|Candidate)\s*\d*[:\s]+(.*?)(?=(?:Composition|Option|Candidate)\s*\d*[:\s]|DESIGN RATIONALE|$)",
        text, re.IGNORECASE | re.DOTALL
    )
    for block in blocks[:3]:
        comp = {}
        oxides = re.findall(r"([A-Z][a-zA-Z0-9]+(?:O\d*)?)\s*[:=]\s*(\d+\.?\d*)\s*(?:wt%|%|)", block)
        for oxide, pct in oxides:
            comp[oxide] = float(pct)
        if comp:
            compositions.append(comp)
    return compositions


def _parse_section(text: str, *section_names) -> list:
    """Extract bullet points from a named section."""
    import re
    for name in section_names:
        match = re.search(
            rf"(?:{name})[:\s]*(.*?)(?:\n\n|\Z)",
            text, re.IGNORECASE | re.DOTALL
        )
        if match:
            lines = match.group(1).strip().split("\n")
            items = []
            for line in lines:
                line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line).strip()
                if len(line) > 10:
                    items.append(line)
            if items:
                return items[:6]
    return []


def _parse_section_text(text: str, *section_names) -> str:
    """Extract a full text section by name."""
    import re
    for name in section_names:
        match = re.search(
            rf"(?:{name})[:\s]*(.*?)(?:\n\n[A-Z]|\Z)",
            text, re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(1).strip()[:800]
    return ""
