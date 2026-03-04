"""
Glass Expert AI — Troubleshoot Router
POST /api/v1/troubleshoot
Given a defect description, returns root causes and corrective actions.
"""
import os
import time
from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models.schemas import (
    TroubleshootRequest, TroubleshootResponse,
    SourceChunk, RootCause, CorrectiveAction
)

router = APIRouter()

# Common glass defect classifications
DEFECT_KEYWORDS = {
    "bubble": "Gaseous Inclusion (Seed/Blister)",
    "seed": "Gaseous Inclusion (Seed)",
    "blister": "Gaseous Inclusion (Blister)",
    "stone": "Solid Inclusion (Stone/Crystallite)",
    "cord": "Compositional Inhomogeneity (Cord/Striae)",
    "striae": "Compositional Inhomogeneity (Striae)",
    "knot": "Solid Inclusion (Refractory Knot)",
    "crack": "Mechanical Fracture / Thermal Stress",
    "haze": "Surface Defect (Haze/Bloom)",
    "warp": "Dimensional Defect (Warp/Bow)",
    "scratch": "Surface Defect (Scratch/Score)",
    "devitrification": "Crystallization Defect (Devitrification)",
    "crystallization": "Crystallization Defect",
    "colour": "Optical Defect (Colour/Tint)",
    "color": "Optical Defect (Colour/Tint)",
    "distortion": "Optical Defect (Optical Distortion)",
    "breakage": "Mechanical Failure (Breakage)",
    "stress": "Residual Stress / Annealing Defect",
}


def _classify_defect(description: str) -> str:
    """Classify the defect type from the description."""
    desc_lower = description.lower()
    for keyword, classification in DEFECT_KEYWORDS.items():
        if keyword in desc_lower:
            return classification
    return "Unclassified Glass Defect"


def _build_troubleshoot_query(request: TroubleshootRequest) -> str:
    """Build a retrieval query from the troubleshoot request."""
    parts = [request.defect_description]
    if request.glass_type:
        parts.append(request.glass_type)
    if request.process_stage:
        parts.append(f"defect at {request.process_stage}")
    parts.append("root cause corrective action")
    return " ".join(parts)


def _build_troubleshoot_prompt(request: TroubleshootRequest, context: str, defect_class: str) -> str:
    """Build the LLM prompt for troubleshooting."""
    return f"""You are Glass Expert AI. Troubleshoot the following glass manufacturing defect.

KNOWLEDGE BASE CONTEXT:
{context}

DEFECT REPORT:
- Description: {request.defect_description}
- Defect Classification: {defect_class}
- Process Stage: {request.process_stage or "Not specified"}
- Glass Type: {request.glass_type or "Not specified"}
- Additional Context: {request.additional_context or "None"}

Provide a structured troubleshooting analysis:

1. DEFECT CLASSIFICATION: Confirm or refine the defect type with technical definition

2. ROOT CAUSES: List the most likely root causes in order of likelihood (high/medium/low). For each:
   - State the cause clearly
   - Explain the mechanism by which it produces this defect
   - Indicate likelihood: HIGH / MEDIUM / LOW

3. CORRECTIVE ACTIONS: For each root cause, provide specific corrective actions with priority:
   - IMMEDIATE: Actions to take right now to stop the defect
   - SHORT-TERM: Process adjustments within days/weeks
   - LONG-TERM: Systematic improvements

4. PREVENTIVE MEASURES: List 3-5 preventive measures to avoid recurrence

5. RELEVANT STANDARDS: List any applicable ASTM, ISO, or EN standards for this defect type

Be specific with process parameters (temperatures, times, concentrations) where the knowledge base provides them."""


@router.post("/troubleshoot", response_model=TroubleshootResponse)
async def troubleshoot_defect(request: TroubleshootRequest):
    """
    Troubleshoot a glass manufacturing defect.

    Provide a description of the defect observed. Optionally include:
    - `process_stage`: where in the process the defect appears
    - `glass_type`: type of glass being produced
    - `additional_context`: recent process changes or observations

    Returns root causes ranked by likelihood, corrective actions by priority,
    preventive measures, and relevant standards.
    """
    from retrieval.retriever import retrieve_with_auto_language, format_context

    start_time = time.time()

    # ── Step 1: Classify defect and build query ────────────────────────────────
    defect_class = _classify_defect(request.defect_description)
    query = _build_troubleshoot_query(request)

    # ── Step 2: Retrieve relevant knowledge ───────────────────────────────────
    try:
        chunks, _ = retrieve_with_auto_language(query=query, top_k=8)
    except Exception as e:
        logger.error(f"Troubleshoot retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    retrieval_time_ms = (time.time() - start_time) * 1000
    context_block = format_context(chunks) if chunks else "No relevant knowledge base entries found for this defect type."

    # ── Step 3: Generate troubleshooting analysis with LLM ────────────────────
    prompt = _build_troubleshoot_prompt(request, context_block, defect_class)

    try:
        from retrieval.llm import _call_openai_compatible
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            analysis_text = await _call_openai_compatible(
                base_url="https://api.openai.com/v1",
                api_key=openai_key,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                system_prompt=(
                    "You are Glass Expert AI, a highly specialized glass manufacturing "
                    "troubleshooting expert with deep knowledge of defect analysis, "
                    "root cause investigation, and corrective actions."
                ),
                user_message=prompt,
                temperature=0.2,
                max_tokens=2000,
            )
            model_used = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        else:
            analysis_text = "LLM not available. Retrieved context:\n\n" + context_block
            model_used = "retrieval-only"
    except Exception as e:
        logger.error(f"Troubleshoot LLM error: {e}")
        analysis_text = "LLM generation failed. Retrieved context:\n\n" + context_block
        model_used = "retrieval-only"

    # ── Step 4: Parse structured output ───────────────────────────────────────
    root_causes = _parse_root_causes(analysis_text)
    corrective_actions = _parse_corrective_actions(analysis_text)
    preventive_measures = _parse_list_section(analysis_text, "PREVENTIVE MEASURES", "PREVENTIVE")
    relevant_standards = _parse_list_section(analysis_text, "RELEVANT STANDARDS", "STANDARDS")

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

    return TroubleshootResponse(
        defect_classification=defect_class,
        root_causes=root_causes if root_causes else [
            RootCause(
                cause="See full analysis above",
                likelihood="medium",
                explanation=analysis_text[:500]
            )
        ],
        corrective_actions=corrective_actions if corrective_actions else [
            CorrectiveAction(
                action="See full analysis above for corrective actions",
                priority="immediate",
                expected_outcome="Defect elimination"
            )
        ],
        preventive_measures=preventive_measures if preventive_measures else ["See full analysis for preventive measures"],
        relevant_standards=relevant_standards if relevant_standards else ["ASTM C162 (Glass Terminology)", "ISO 11485 (Glass in Building)"],
        sources=sources,
        model_used=model_used,
        retrieval_time_ms=round(retrieval_time_ms, 2),
    )


def _parse_root_causes(text: str) -> list:
    """Extract root causes with likelihood from LLM output."""
    import re
    causes = []
    # Look for ROOT CAUSES section
    section = re.search(
        r"ROOT CAUSES?[:\s]*(.*?)(?:\n\n[A-Z3-9]|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if not section:
        return causes

    content = section.group(1)
    # Split by numbered items or bullet points
    items = re.split(r"\n\s*(?:\d+[\.\)]\s*|-\s*|\*\s*)", content)
    for item in items:
        item = item.strip()
        if len(item) < 15:
            continue
        # Detect likelihood
        likelihood = "medium"
        if re.search(r"\bHIGH\b", item, re.IGNORECASE):
            likelihood = "high"
        elif re.search(r"\bLOW\b", item, re.IGNORECASE):
            likelihood = "low"

        # Split cause from explanation
        lines = item.split("\n")
        cause_line = lines[0].strip()
        explanation = " ".join(l.strip() for l in lines[1:] if l.strip())

        cause_line = re.sub(r"\s*[-–]\s*(HIGH|MEDIUM|LOW)\s*$", "", cause_line, flags=re.IGNORECASE).strip()

        if cause_line:
            causes.append(RootCause(
                cause=cause_line[:200],
                likelihood=likelihood,
                explanation=explanation[:500] if explanation else cause_line,
            ))
        if len(causes) >= 5:
            break
    return causes


def _parse_corrective_actions(text: str) -> list:
    """Extract corrective actions with priority from LLM output."""
    import re
    actions = []
    section = re.search(
        r"CORRECTIVE ACTIONS?[:\s]*(.*?)(?:\n\n[A-Z]|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if not section:
        return actions

    content = section.group(1)
    items = re.split(r"\n\s*(?:\d+[\.\)]\s*|-\s*|\*\s*)", content)
    for item in items:
        item = item.strip()
        if len(item) < 15:
            continue
        # Detect priority
        priority = "short-term"
        if re.search(r"\bIMMEDIATE\b", item, re.IGNORECASE):
            priority = "immediate"
        elif re.search(r"\bLONG.TERM\b", item, re.IGNORECASE):
            priority = "long-term"

        lines = item.split("\n")
        action_line = lines[0].strip()
        outcome = " ".join(l.strip() for l in lines[1:] if l.strip())

        action_line = re.sub(r"\s*[-–]\s*(IMMEDIATE|SHORT.TERM|LONG.TERM)\s*$", "", action_line, flags=re.IGNORECASE).strip()

        if action_line:
            actions.append(CorrectiveAction(
                action=action_line[:300],
                priority=priority,
                expected_outcome=outcome[:300] if outcome else "Defect reduction/elimination",
            ))
        if len(actions) >= 6:
            break
    return actions


def _parse_list_section(text: str, *section_names) -> list:
    """Extract a bullet list from a named section."""
    import re
    for name in section_names:
        match = re.search(
            rf"(?:{name})[:\s]*(.*?)(?:\n\n[A-Z]|\Z)",
            text, re.IGNORECASE | re.DOTALL
        )
        if match:
            lines = match.group(1).strip().split("\n")
            items = []
            for line in lines:
                line = re.sub(r"^[\d\.\-\*\•]+\s*", "", line).strip()
                if len(line) > 8:
                    items.append(line)
            if items:
                return items[:6]
    return []
