"""
Glass Expert AI — Analyze Router
POST /api/v1/analyze
Analyzes a glass composition or sample description and returns predicted properties.
"""
import time
from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from api.auth import get_current_user, UserInToken

from api.models.schemas import (
    AnalyzeRequest, AnalyzeResponse, SourceChunk, PropertyPrediction
)

router = APIRouter()


def _build_analyze_query(request: AnalyzeRequest) -> str:
    """Build a retrieval query from the analyze request."""
    parts = []
    if request.composition:
        oxides = ", ".join(f"{k}: {v}%" for k, v in request.composition.items())
        parts.append(f"glass composition {oxides}")
    if request.description:
        parts.append(request.description)
    if request.properties_of_interest:
        parts.append("properties: " + ", ".join(request.properties_of_interest))
    if not parts:
        raise ValueError("Either composition or description must be provided.")
    return " ".join(parts)


def _build_analyze_prompt(request: AnalyzeRequest, context: str) -> str:
    """Build the LLM prompt for composition analysis."""
    composition_str = ""
    if request.composition:
        composition_str = "\n".join(
            f"  - {oxide}: {pct:.1f} wt%" for oxide, pct in request.composition.items()
        )
        composition_str = f"\nGlass Composition (wt%):\n{composition_str}"

    props_str = ""
    if request.properties_of_interest:
        props_str = f"\nProperties of Interest: {', '.join(request.properties_of_interest)}"

    return f"""You are Glass Expert AI. Analyze the following glass based on the knowledge base context.

KNOWLEDGE BASE CONTEXT:
{context}
{composition_str}
{props_str}
{"Description: " + request.description if request.description else ""}

Provide a structured technical analysis with:
1. COMPOSITION SUMMARY: Brief characterization of the glass type and system
2. PREDICTED PROPERTIES: For each relevant property, give the expected value/range with units and confidence level (high/medium/low). Include: Tg (°C), CTE (ppm/K), density (g/cm³), refractive index, viscosity at working point, chemical durability, and any others relevant to the composition.
3. ANALYSIS: Detailed technical assessment of the composition — network formers, modifiers, intermediates, and their roles
4. RECOMMENDATIONS: Specific suggestions for optimization or caution points

Be precise and cite specific values from the knowledge base where available."""


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_glass(request: AnalyzeRequest, user: UserInToken = Depends(get_current_user)):
    """
    Analyze a glass composition or sample.

    Provide either:
    - `composition`: dict of oxide → weight% (e.g. {"SiO2": 72.0, "Na2O": 14.0})
    - `description`: free-text description of the glass or problem
    - Both together for the most complete analysis
    """
    from retrieval.retriever import retrieve_with_auto_language, format_context
    from retrieval.llm import generate_answer

    if not request.composition and not request.description:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: 'composition' (dict) or 'description' (string)"
        )

    start_time = time.time()

    # ── Step 1: Build retrieval query ─────────────────────────────────────────
    try:
        query = _build_analyze_query(request)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # ── Step 2: Retrieve relevant knowledge ───────────────────────────────────
    try:
        chunks, detected_language = retrieve_with_auto_language(
            query=query,
            top_k=8,  # More chunks for composition analysis
        )
    except Exception as e:
        logger.error(f"Analyze retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    retrieval_time_ms = (time.time() - start_time) * 1000

    # ── Step 3: Generate analysis with LLM ────────────────────────────────────
    context_block = format_context(chunks) if chunks else "No relevant knowledge base entries found."
    prompt = _build_analyze_prompt(request, context_block)

    try:
        from retrieval.llm import _call_openai_compatible
        import os
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            analysis_text = await _call_openai_compatible(
                base_url="https://api.openai.com/v1",
                api_key=openai_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                system_prompt="You are Glass Expert AI, a highly specialized glass science and manufacturing expert.",
                user_message=prompt,
                temperature=0.2,
                max_tokens=1500,
            )
            model_used = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            analysis_text = (
                "LLM not available. Retrieved context:\n\n" + context_block
            )
            model_used = "retrieval-only"
    except Exception as e:
        logger.error(f"Analyze LLM error: {e}")
        analysis_text = "LLM generation failed. Retrieved context:\n\n" + context_block
        model_used = "retrieval-only"

    # ── Step 4: Parse predicted properties from LLM output ───────────────────
    # Extract structured property predictions (best-effort parsing)
    predicted_properties = _extract_properties(analysis_text, request.composition)

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

    composition_summary = None
    if request.composition:
        dominant = max(request.composition, key=request.composition.get)
        composition_summary = (
            f"{len(request.composition)}-component glass system dominated by "
            f"{dominant} ({request.composition[dominant]:.1f} wt%)"
        )

    return AnalyzeResponse(
        composition_summary=composition_summary,
        analysis=analysis_text,
        predicted_properties=predicted_properties,
        recommendations=_extract_recommendations(analysis_text),
        sources=sources,
        model_used=model_used,
        retrieval_time_ms=round(retrieval_time_ms, 2),
    )


def _extract_properties(text: str, composition: dict) -> list:
    """Best-effort extraction of property predictions from LLM output."""
    import re
    properties = []

    # Common property patterns to look for
    patterns = [
        (r"Tg[:\s]+([~≈]?\s*\d+[\-–]\d+|\d+)\s*°?C", "Glass Transition Temperature (Tg)", "°C"),
        (r"CTE[:\s]+([~≈]?\s*\d+\.?\d*[\-–]\d*\.?\d*)\s*(?:×\s*10[-−]?7|ppm)", "Thermal Expansion Coefficient (CTE)", "×10⁻⁷/K"),
        (r"density[:\s]+([~≈]?\s*\d+\.?\d*)\s*g/cm", "Density", "g/cm³"),
        (r"refractive index[:\s]+([~≈]?\s*\d+\.\d+)", "Refractive Index (nd)", ""),
    ]

    for pattern, prop_name, unit in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            properties.append(PropertyPrediction(
                property_name=prop_name,
                value=match.group(1).strip(),
                unit=unit,
                confidence="medium",
                source="LLM analysis",
            ))

    # If no properties extracted, add placeholders based on composition
    if not properties and composition:
        sio2 = composition.get("SiO2", composition.get("sio2", 0))
        if sio2 > 60:
            properties.append(PropertyPrediction(
                property_name="Glass Type",
                value="Silicate glass" if sio2 > 90 else "Modified silicate glass",
                unit=None,
                confidence="high",
                source="Composition analysis",
            ))

    return properties


def _extract_recommendations(text: str) -> list:
    """Extract recommendation bullet points from LLM output."""
    import re
    recommendations = []
    # Look for numbered or bulleted recommendations section
    rec_section = re.search(
        r"(?:RECOMMENDATIONS?|SUGGESTIONS?)[:\s]*(.*?)(?:\n\n|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if rec_section:
        lines = rec_section.group(1).strip().split("\n")
        for line in lines:
            line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line).strip()
            if len(line) > 10:
                recommendations.append(line)
    if not recommendations:
        recommendations = ["See full analysis above for detailed recommendations."]
    return recommendations[:5]  # Max 5 recommendations
