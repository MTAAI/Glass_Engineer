"""
Glass Expert AI — LLM Integration
Supports:
  - Local vLLM (Qwen 2.5-14B fine-tuned) via OpenAI-compatible API
  - OpenAI GPT-4o-mini (fallback when local model unavailable)
  - Retrieval-only mode (no LLM, returns formatted context)
"""
import os
import re
import json
from datetime import datetime, timezone
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Singleton client cache — avoids creating a new connection per request
_client_cache: dict = {}

# ── Fallback governance: structured logging ──────────────────────────────────
_FALLBACK_LOG_PATH = os.getenv("FALLBACK_LOG_PATH", "logs/fallback_governance.jsonl")


def _log_fallback_event(
    reason: str,
    question: str,
    language: str,
    local_error: str = "",
    model_used: str = "gpt-4o-mini",
):
    """Log every OpenAI fallback call for governance/auditing.
    Writes structured JSONL so it's easy to grep, count, and alert on.
    """
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "openai_fallback",
        "reason": reason,
        "language": language,
        "model_used": model_used,
        "question_preview": question[:120],
        "local_error": local_error[:200] if local_error else "",
    }
    logger.warning(
        f"FALLBACK → {model_used} | reason={reason} | lang={language} | q='{question[:60]}...'"
    )
    try:
        os.makedirs(os.path.dirname(_FALLBACK_LOG_PATH) or "logs", exist_ok=True)
        with open(_FALLBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"Could not write fallback log: {e}")

# ── System prompts ─────────────────────────────────────────────────────────
# Full prompt for cloud models (GPT-4o-mini) — detailed instructions
SYSTEM_PROMPT_EN = """You are Glass Expert AI, a highly specialized assistant for glass scientists and manufacturing engineers with PhD-level expertise.

CRITICAL RULES:
- Answer ONLY using information from the KNOWLEDGE BASE CONTEXT provided below.
- Do NOT fabricate values, compositions, temperatures, or any data not present in the context.
- If the context lacks specific information, state "the provided context does not specify..." rather than guessing.
- Extract and quote exact numerical values, ranges, and units directly from the context.
- When multiple sources provide data on the same topic, SYNTHESIZE them — combine complementary details and note any conflicts between sources.
- Every numerical claim (temperature, composition, property value) MUST be traceable to a specific [Source N].

ANSWER STRUCTURE:
1. Lead with a direct, concise answer to the question (1-2 sentences).
2. When the question is about a well-known glass type (e.g., borosilicate, soda-lime, aluminosilicate, lead glass, E-glass),
   ALWAYS state the standard/commercial property value first (e.g., "Standard borosilicate glass such as Pyrex has a Tg of ~560°C (820-830 K)").
   Then provide composition-specific details from the context as supplementary information.
3. Follow with detailed technical explanation using numbered points.
4. Include specific compositions (e.g., 72% SiO2, 14% Na2O) and property values FROM the context.
5. For processes, list ALL stages with their specific temperature ranges and conditions.
6. Reference named scientists, equations (e.g., Abbe number V=(n_d-1)/(n_F-n_C)), and standards (ISO, ASTM) when they appear in the context.
7. Use precise technical terms: network formers, network modifiers, bridging oxygen (BO), non-bridging oxygen (NBO), coordination number, fining agents, devitrification, etc.
8. Cite sources as [Source 1], [Source 2], etc. for every key fact.
9. End with a brief summary if the answer covers multiple aspects.
10. Keep answers focused, quantitative, and specific — never give vague generalizations.

SOURCE PRIORITY: Textbook definitions, standards, and general knowledge sources should take precedence over
specific simulation data or individual composition measurements when answering general questions."""

SYSTEM_PROMPT_FA = """You are Glass Expert AI, a highly specialized assistant for glass scientists and manufacturing engineers with PhD-level expertise. You MUST answer entirely in Persian/Farsi.

CRITICAL RULES:
- Answer ONLY using information from the KNOWLEDGE BASE CONTEXT provided below.
- Do NOT fabricate values, compositions, temperatures, or any data not present in the context.
- If the context lacks specific information, state in Farsi: "متن ارائه شده این اطلاعات را مشخص نمی‌کند" rather than guessing.
- Extract and quote exact numerical values, ranges, and units directly from the context.
- When multiple sources provide data on the same topic, SYNTHESIZE them — combine complementary details and note any conflicts between sources.
- Every numerical claim (temperature, composition, property value) MUST be traceable to a specific [Source N] / [منبع N].

ANSWER STRUCTURE (write everything in Farsi):
1. Lead with a direct, concise answer to the question (1-2 sentences in Farsi).
2. For well-known glass types, state the standard/commercial property value first, then context-specific details.
3. Follow with detailed technical explanation using numbered points.
4. Include specific compositions (e.g., 72% SiO2, 14% Na2O) and property values FROM the context.
5. For processes, list ALL stages with their specific temperature ranges and conditions.
6. Cite sources as [منبع ۱], [منبع ۲], etc. for every key fact.
7. Keep answers focused, quantitative, and specific — never give vague generalizations.

LANGUAGE: Numbers, chemical formulas (SiO2, Na2O), units (°C, MPa), and abbreviations may remain in their original form. Everything else MUST be in Farsi.

SOURCE PRIORITY: Textbook definitions, standards, and general knowledge sources should take precedence over specific simulation data."""

# ── Local model prompt — matches Qwen 14B training format ────────────────────
# The Qwen 14B model was fine-tuned on plain Q&A pairs with THIS system prompt.
# Using the EXACT training system prompt is critical for good generation.
LOCAL_SYSTEM_PROMPT = """You are Glass Expert AI, a highly specialized assistant trained on thousands of glass science research papers, textbooks, and material databases. You provide accurate, detailed, and technically precise answers about glass composition, properties, manufacturing processes, defects, characterization, and applications. Always cite relevant glass science principles in your answers."""

# Farsi-specific local prompt: English instructions (better instruction-following)
# with explicit Farsi output rule. Research shows Qwen 2.5 follows English
# instructions more reliably even when outputting in other languages.
LOCAL_SYSTEM_PROMPT_FA = """You are Glass Expert AI, a highly specialized assistant trained on thousands of glass science research papers, textbooks, and material databases. You provide accurate, detailed, and technically precise answers about glass composition, properties, manufacturing processes, defects, characterization, and applications.

CRITICAL RULES:
1. Answer ONLY from the reference information provided. Do NOT use your general training knowledge.
2. You MUST write your ENTIRE response in Persian/Farsi.
3. Numbers, chemical formulas (SiO2, Na2O), units (°C, MPa), and standard abbreviations may remain in their original form.
4. Extract exact numerical values directly from the reference text.
5. If the reference text does not contain the information, say in Farsi: "اطلاعات مرجع ارائه شده شامل این جزئیات نیست".
6. Structure your answer: brief definition, then numbered key points with specific values."""


def _strip_rag_formatting(context: str) -> str:
    """
    Simplify RAG metadata formatting for the local Qwen 14B model.

    Keeps source titles for grounding (so the model knows what it's citing)
    but strips verbose metadata (Type:, Language:, Relevance:) that adds
    noise without helping answer quality.

    Input format (from format_context_for_llm):
        KNOWLEDGE BASE CONTEXT:
        ==================================================
        [Source 1] Title (Type: paper | Language: EN | Relevance: 83%)
        actual content here...
        ----------------------------------------

    Output format (structured but clean):
        [Source 1] Title
        actual content here...

    """
    if not context:
        return context

    lines = context.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip header lines
        if stripped == "KNOWLEDGE BASE CONTEXT:":
            continue
        # Skip separator lines (=== or ---)
        if stripped and all(c in "=-" for c in stripped) and len(stripped) > 5:
            continue
        # Clean [Source N] lines: keep title, strip metadata in parentheses
        source_match = re.match(r'^(\[Source \d+\]\s+.+?)\s*\(Type:.*\)', stripped)
        if source_match:
            clean_lines.append(source_match.group(1))
            continue
        # Keep [Source N] lines that don't have metadata
        if re.match(r'^\[Source \d+\]', stripped):
            clean_lines.append(stripped)
            continue
        # Keep everything else (the actual content)
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def _is_degenerate(text: str) -> bool:
    """Detect repetitive/nonsensical LLM output that should trigger fallback.

    Tuned for Qwen 14B — less aggressive than the Llama 8B version.
    Qwen legitimately says 'the provided text' when context is insufficient,
    which is a valid answer, not degenerate output.
    """
    if not text or len(text.split()) < 5:
        return True

    words = text.split()
    from collections import Counter

    # Check for excessive repetition: if any 3-word phrase repeats 5+ times
    if len(words) > 20:
        phrases = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        most_common = Counter(phrases).most_common(1)
        if most_common and most_common[0][1] >= 5:
            return True

    # Check for output that just echoes context metadata
    lower = text.lower()
    if lower.count("[source n]") >= 2 or lower.count("source_type") >= 2:
        return True

    # Check for pure meta-commentary (model ONLY describes the text, never answers)
    # Only flag if the ENTIRE response is meta-commentary (not just a prefix)
    meta_only_patterns = [
        "the text is written",
        "the text is a reference",
        "this text describes",
        "this response is referenced",
    ]
    if len(words) < 30 and any(p in lower for p in meta_only_patterns):
        return True

    # Check for truncated/fragmented output (no complete sentence)
    if len(text) < 50 and "." not in text and ":" not in text and "،" not in text:
        return True

    return False


def _is_english_response(text: str) -> bool:
    """Detect if a response is primarily in English when Farsi was expected.

    Uses a simple heuristic: count ASCII letter words vs total words.
    Farsi text uses Persian/Arabic script; English uses Latin script.
    Chemical formulas (SiO2, Na2O) and numbers are excluded from the check.
    """
    if not text or len(text) < 20:
        return False

    words = text.split()
    if len(words) < 5:
        return False

    ascii_words = 0
    total_words = 0
    for w in words:
        # Skip numbers, formulas, units, citations
        cleaned = w.strip(".,;:()[]{}«»-–—/\\")
        if not cleaned:
            continue
        # Skip pure numbers and chemical formulas (e.g., SiO2, Na2O, 500°C)
        if re.match(r'^[\d.,%°±]+$', cleaned):
            continue
        if re.match(r'^[A-Z][a-z]?\d', cleaned):  # chemical formula
            continue
        if len(cleaned) <= 2:  # skip short tokens (units like °C, MPa)
            continue

        total_words += 1
        # Check if word is ASCII (English/Latin)
        if all(ord(c) < 128 for c in cleaned):
            ascii_words += 1

    if total_words < 5:
        return False

    english_ratio = ascii_words / total_words
    # If more than 60% of meaningful words are English, it's an English response
    return english_ratio > 0.60


def _compress_history(messages: list) -> list:
    """Compress conversation history to fit more turns in the token budget.

    Strategy:
      - Last 4 messages (2 turns): keep full content
      - Older messages: truncate assistant responses to first 150 chars
        (keeps the gist without burning tokens on full RAG answers)
      - Strip any KNOWLEDGE BASE CONTEXT blocks from history
        (they're from previous queries, not relevant now)
    """
    if len(messages) <= 4:
        return messages

    compressed = []
    cutoff = len(messages) - 4  # keep last 4 full

    for i, msg in enumerate(messages):
        content = msg.get("content", "")

        # Strip old context blocks from all history messages
        if "KNOWLEDGE BASE CONTEXT:" in content:
            # Extract just the question part
            parts = content.split("QUESTION:")
            if len(parts) > 1:
                content = parts[-1].strip()
            else:
                # Try to find the question after the context block
                lines = content.split("\n")
                content = " ".join(
                    l for l in lines
                    if not l.startswith("=") and not l.startswith("-" * 10)
                    and not l.startswith("[Source") and "KNOWLEDGE BASE" not in l
                )[:300]

        if i < cutoff and msg.get("role") == "assistant":
            # Summarize older assistant responses
            content = content[:150].rsplit(" ", 1)[0] + "..." if len(content) > 150 else content

        compressed.append({"role": msg["role"], "content": content})

    return compressed


async def generate_answer(
    question: str,
    context: str,
    language: str = "en",
    conversation_history: list | None = None,
    user_memory: list | None = None,
) -> tuple[str, str]:
    """
    Generate an expert answer using the configured LLM.

    Strategy:
    1. Try local model first with PLAIN TEXT context (matches training format).
       The local model was fine-tuned on plain Q&A, not RAG-formatted context.
    2. If local model fails or produces garbage, fall back to OpenAI GPT-4o-mini
       with FULL RAG-formatted context (cloud models handle structured prompts well).

    Args:
        conversation_history: List of prior messages as dicts with 'role' and 'content'.
                              Capped at 10 messages (5 turns) to manage token budget.
        user_memory: List of user memory dicts with 'key' and 'value' for personalization.

    Returns:
        Tuple of (answer_text, model_name_used)
    """
    llm_url = os.getenv("LLM_BASE_URL", "")
    llm_model = os.getenv("LLM_MODEL", "glass-expert-qwen14b")
    llm_api_key = os.getenv("LLM_API_KEY", "token-glass-ai")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "800"))

    # Full system prompts for cloud models
    cloud_system_prompt = SYSTEM_PROMPT_FA if language == "fa" else SYSTEM_PROMPT_EN

    # Inject user memory/preferences into system prompt
    if user_memory:
        memory_lines = []
        for mem in user_memory:
            memory_lines.append(f"- {mem['key']}: {mem['value']}")
        if memory_lines:
            memory_block = "\n\nUser context (personalization):\n" + "\n".join(memory_lines)
            cloud_system_prompt += memory_block

    # Add conversation continuity instruction when history is present
    if conversation_history:
        continuity = (
            "\n\nYou are in an ongoing conversation. Prior messages are provided for context. "
            "Use them to understand follow-up questions (e.g., 'what about its thermal properties?' "
            "refers to the glass type discussed earlier). Maintain continuity but always ground "
            "answers in the KNOWLEDGE BASE CONTEXT — do not repeat previous answers verbatim."
        )
        cloud_system_prompt += continuity

    # Full user messages for cloud models (keep RAG formatting — GPT handles it well)
    if language == "fa":
        cloud_user_message = f"""KNOWLEDGE BASE CONTEXT:
{context}

QUESTION: {question}

Provide a precise, technical answer based strictly on the knowledge base context above.
- Extract EXACT numerical values, temperatures, compositions directly from the context.
- Reference sources as [منبع ۱], [منبع ۲], etc. for every key fact.
- Do NOT fabricate any data not present in the context.
- Write your ENTIRE answer in Persian/Farsi. Only numbers, chemical formulas, and units may remain in English."""
    else:
        cloud_user_message = f"""KNOWLEDGE BASE CONTEXT:
{context}

QUESTION: {question}

Provide a precise, technical answer based strictly on the knowledge base context above. Reference sources by their [Source N] numbers."""

    # Smart history compression: keep recent turns full, summarize older ones
    history = _compress_history((conversation_history or [])[-10:])

    # ── Try local LLM (Qwen 14B — handles both English and Farsi) ──────────
    local_failed = False
    plain_context = _strip_rag_formatting(context)

    # Context limit: balance quality vs latency
    # 8000 chars (~2000 tokens) is enough for 6-8 good sources
    # Larger context = more prompt tokens = slower inference
    local_context_limit = 8000  # chars (~2000 tokens)
    if len(plain_context) > local_context_limit:
        plain_context = plain_context[:local_context_limit]
        logger.debug(f"Truncated context for local model: {len(context)} → {local_context_limit} chars")

    if language == "fa":
        local_user_message = f"""Reference information:
{plain_context}

Question: {question}

IMPORTANT: Your response MUST be written entirely in Persian/Farsi (فارسی). Do NOT write in English.

Instructions: Answer STRICTLY from the reference information above.
- Extract EXACT numerical values, temperatures, compositions, and property data directly from the reference text.
- Do NOT use your general knowledge — ONLY use data found in the reference text above.
- If the reference text does not contain specific information, say: "اطلاعات مرجع ارائه شده شامل این جزئیات نیست".
- Cite specific data points with their values and units.
- Structure your answer: brief definition in Farsi, then numbered key points with specific values from the references.
- ALL text must be in Farsi. Only numbers, formulas (SiO2, Na2O), and units (°C, MPa) may remain in English.

پاسخ خود را به فارسی بنویسید:"""
    else:
        local_user_message = f"""Reference information:
{plain_context}

Question: {question}

Instructions: Answer STRICTLY from the reference information above.
- Extract EXACT numerical values, temperatures, compositions, and property data directly from the reference text.
- Do NOT use your general knowledge — ONLY use data found in the reference text above.
- If the reference text does not contain specific information, say "the provided references do not specify..." rather than guessing.
- Cite specific data points with their values and units.
- Structure your answer: brief definition, then numbered key points with specific values from the references."""

    local_url = llm_url if llm_url else "http://localhost:8000/v1"
    try:
        # Local model: use shorter history (4 messages = 2 turns) to save tokens
        local_history = history[-4:] if history else None
        # Local model: 1024 tokens is enough for detailed answer
        # Higher values increase latency linearly (each token = ~50ms on 14B)
        local_max_tokens = max(max_tokens, 1024)
        answer = await _call_openai_compatible(
            base_url=local_url,
            api_key=llm_api_key,
            model=llm_model,
            system_prompt=LOCAL_SYSTEM_PROMPT_FA if language == "fa" else LOCAL_SYSTEM_PROMPT,
            user_message=local_user_message,
            temperature=temperature,
            max_tokens=local_max_tokens,
            conversation_history=local_history,
        )
        # Quality gate: detect degenerate output (repetitive/nonsensical)
        if _is_degenerate(answer):
            logger.warning(f"Local model produced degenerate output ({len(answer)} chars): {answer[:200]!r}")
            _log_fallback_event("degenerate_output", question, language, local_error=answer[:200])
            local_failed = True
        # Language gate: if Farsi was requested but answer is in English, retry once
        elif language == "fa" and _is_english_response(answer):
            logger.warning("Local model answered in English for Farsi query — retrying with stronger instruction")
            try:
                retry_message = f"""سوال: {question}

اطلاعات مرجع:
{plain_context[:4000]}

دستورالعمل مهم: پاسخ را کاملاً به زبان فارسی بنویسید. هیچ جمله‌ای به انگلیسی ننویسید.
مقادیر عددی و فرمول‌های شیمیایی می‌توانند به شکل اصلی باقی بمانند.

پاسخ فارسی:"""
                answer2 = await _call_openai_compatible(
                    base_url=local_url, api_key=llm_api_key, model=llm_model,
                    system_prompt=LOCAL_SYSTEM_PROMPT_FA,
                    user_message=retry_message,
                    temperature=temperature, max_tokens=local_max_tokens,
                )
                if not _is_degenerate(answer2) and not _is_english_response(answer2):
                    logger.info("Farsi retry succeeded")
                    return answer2, llm_model
                else:
                    logger.warning("Farsi retry still in English — using original answer")
                    return answer, llm_model  # still return the English answer rather than fallback
            except Exception:
                return answer, llm_model  # return original answer on retry failure
        else:
            return answer, llm_model
    except Exception as e:
        logger.warning(f"Local LLM at {local_url} not available: {e}")
        _log_fallback_event("local_unavailable", question, language, local_error=str(e))
        local_failed = True

    # ── Fallback: OpenAI GPT (local model failure) ────────────────────────────
    if local_failed:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            try:
                logger.info("Using OpenAI fallback due to local model failure")
                answer = await _call_openai_compatible(
                    base_url="https://api.openai.com/v1",
                    api_key=openai_key,
                    model="gpt-4o-mini",
                    system_prompt=cloud_system_prompt,
                    user_message=cloud_user_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    conversation_history=history,
                )
                return answer, "gpt-4o-mini (fallback)"
            except Exception as e:
                logger.warning(f"OpenAI fallback failed: {e}")

    # ── Final fallback ────────────────────────────────────────────────────────
    logger.warning("No LLM available. Returning retrieved context as answer.")
    raise RuntimeError("No LLM backend available")


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((TimeoutError, ConnectionError, OSError)),
    reraise=True,
)
async def _call_openai_compatible(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float,
    max_tokens: int,
    conversation_history: list | None = None,
) -> str:
    """Call any OpenAI-compatible API (vLLM, OpenAI, etc.).
    Retries up to 3 times on network errors with exponential backoff."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    cache_key = f"{base_url}|{api_key}"
    if cache_key not in _client_cache:
        _client_cache[cache_key] = AsyncOpenAI(base_url=base_url, api_key=api_key)
    client = _client_cache[cache_key]

    # Build messages: system → prior conversation turns → current user message
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        for msg in conversation_history:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # Qwen 2.5 official recommended sampling params (from generation_config.json)
    # MUST be passed explicitly — vLLM defaults are NOT suitable for Qwen
    extra = {}
    if "localhost" in base_url or "127.0.0.1" in base_url:
        extra["extra_body"] = {
            "repetition_penalty": 1.05,  # official Qwen 2.5 value — higher breaks lists/tables
            "top_k": 20,                 # official Qwen 2.5 value
        }
        extra["top_p"] = 0.8            # official Qwen 2.5 value

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=60.0,
        **extra,
    )

    answer = response.choices[0].message.content.strip()

    # Log token usage for debugging
    if response.usage:
        logger.debug(
            f"LLM tokens — prompt: {response.usage.prompt_tokens}, "
            f"completion: {response.usage.completion_tokens}, "
            f"total: {response.usage.total_tokens}"
        )

    return answer
