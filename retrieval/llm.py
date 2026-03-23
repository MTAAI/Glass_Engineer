"""
Glass Expert AI — LLM Integration
Supports:
  - Local vLLM (Llama-3-8B) via OpenAI-compatible API
  - OpenAI GPT-4 (fallback or cloud mode)
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

SYSTEM_PROMPT_FA = """شما Glass Expert AI هستید، یک دستیار تخصصی با تخصص سطح دکترا برای دانشمندان شیشه و مهندسان تولید.

قواعد حیاتی:
- فقط بر اساس متن پایگاه دانش ارائه شده پاسخ دهید — هرگز اطلاعات جعلی ارائه ندهید
- مقادیر، فرمول‌ها و ترکیبات دقیق را مستقیماً از متن استخراج و نقل کنید
- اگر متن اطلاعات کافی ندارد، بگویید «متن ارائه شده این اطلاعات را مشخص نمی‌کند» — حدس نزنید

قالب‌بندی:
1. با تعریف یا پاسخ مستقیم شروع کنید
2. وقتی سوال درباره یک نوع شیشه شناخته‌شده است (مثل بوروسیلیکات، سودا-لایم، آلومینوسیلیکات)،
   ابتدا مقدار استاندارد/تجاری را ذکر کنید (مثلاً «شیشه بوروسیلیکات استاندارد مانند Pyrex دمای Tg حدود ۵۶۰ درجه سانتیگراد دارد»).
   سپس داده‌های ترکیبات خاص از متن را به عنوان اطلاعات تکمیلی ارائه دهید.
3. از لیست شماره‌دار برای فرآیندها، قوانین و روش‌ها استفاده کنید
4. ترکیبات خاص (مثلاً ۷۲٪ SiO2) و مقادیر خواص را از متن ذکر کنید
5. نام سند منبع را در کروشه ذکر کنید، مثلاً [منبع ۱]
6. اگر منابع مختلف اطلاعات متناقضی دارند، تناقض را ذکر کنید
7. مقادیر عددی را با واحد و شرایط (دما، فشار، ترکیب) ذکر کنید

اولویت منابع: تعاریف کتاب درسی، استانداردها و منابع دانش عمومی بر داده‌های شبیه‌سازی خاص اولویت دارند."""

# ── Local model prompt — matches training format ────────────────────────────
# The local 8B model was fine-tuned on plain Q&A pairs with THIS system prompt.
# Using the EXACT training system prompt is critical for good generation.
LOCAL_SYSTEM_PROMPT = """You are Glass Expert AI, a highly specialized assistant trained on thousands of glass science research papers, textbooks, and material databases. You provide accurate, detailed, and technically precise answers about glass composition, properties, manufacturing processes, defects, characterization, and applications.

RULES:
- Answer ONLY using the reference information provided. Do NOT fabricate data.
- Lead with the standard/textbook definition or value first, then add specific details.
- Include exact numerical values (temperatures, compositions, percentages) from the reference.
- If the reference lacks information, say so — do not guess.
- Use proper glass science terminology: network formers, network modifiers, bridging oxygen (BO), non-bridging oxygen (NBO), fining agents, devitrification, etc.
- Keep answers focused, quantitative, and technically precise."""


def _strip_rag_formatting(context: str) -> str:
    """
    Strip ALL RAG metadata to produce pure plain text for the local model.

    The v2 model was trained on pure Q&A — no [Source N], no metadata, no
    separators. ANY formatting the model hasn't seen during training confuses
    it into meta-commentary. Strip EVERYTHING except the actual content text.

    Input format (from format_context_for_llm):
        KNOWLEDGE BASE CONTEXT:
        ==================================================
        [Source 1] Title (Type: paper | Language: EN | Relevance: 83%)
        actual content here...
        ----------------------------------------

    Output format (pure plain text):
        actual content here...
    """
    if not context:
        return context

    lines = context.split("\n")
    clean_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip empty lines at boundaries
        if not stripped:
            # Keep paragraph breaks but don't stack them
            if clean_lines and clean_lines[-1].strip():
                clean_lines.append("")
            continue
        # Skip header lines
        if stripped == "KNOWLEDGE BASE CONTEXT:":
            continue
        # Skip separator lines (=== or ---)
        if all(c in "=-" for c in stripped) and len(stripped) > 5:
            continue
        # Skip [Source N] lines entirely — model has never seen these
        if re.match(r'^\[Source \d+\]', stripped):
            continue
        # Skip lines that are just metadata
        if stripped.startswith("Type:") or stripped.startswith("Language:") or stripped.startswith("Relevance:"):
            continue
        # Keep everything else (the actual content)
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()


def _is_degenerate(text: str) -> bool:
    """Detect repetitive/nonsensical LLM output that should trigger fallback."""
    if not text or len(text.split()) < 8:
        return True

    words = text.split()
    from collections import Counter

    # Check for excessive repetition: if any 3-word phrase repeats 4+ times
    if len(words) > 15:
        phrases = [" ".join(words[i:i+3]) for i in range(len(words) - 2)]
        most_common = Counter(phrases).most_common(1)
        if most_common and most_common[0][1] >= 4:
            return True

    # Check for output that just echoes the question or context metadata
    lower = text.lower()
    if lower.count("[source n]") >= 2 or lower.count("source_type") >= 2:
        return True

    # Check for meta-commentary patterns (model describes the text instead of answering)
    # Only flag if the answer STARTS with meta-commentary (not if it appears mid-answer)
    meta_patterns = [
        "the text is written",
        "the text is a reference",
        "this text describes",
        "the passage discusses",
        "the above text",
        "this response is referenced",
    ]
    # Only check first 150 chars — meta-commentary at the start is bad,
    # but "the provided text mentions..." mid-answer is fine
    first_part = lower[:150]
    if any(p in first_part for p in meta_patterns):
        return True

    # Check for truncated/fragmented output (no complete sentence)
    if len(text) < 80 and "." not in text and ":" not in text and "،" not in text:
        return True

    return False


def _compress_history(messages: list) -> list:
    """Compress conversation history to fit more turns in the token budget.

    Strategy (3-tier):
      - Last 6 messages (3 turns): keep FULL content — immediate context matters most
      - Middle messages (turns 4-8): keep user questions full, assistant answers trimmed to 300 chars
      - Oldest messages (turns 9+): keep user questions only (drop assistant responses)
      - Strip any KNOWLEDGE BASE CONTEXT / Reference information blocks from all history
        (they're from previous queries, not relevant now)

    This gives the model:
      - Full context of the last 3 exchanges
      - Topic awareness from older questions
      - Total token budget stays manageable (~2000 tokens for history)
    """
    if len(messages) <= 6:
        # Short history — keep everything, just strip old context
        return [
            {"role": m["role"], "content": _clean_history_content(m.get("content", ""))}
            for m in messages
        ]

    compressed = []
    full_cutoff = len(messages) - 6     # last 6 kept full
    middle_cutoff = len(messages) - 16  # middle tier: trimmed

    for i, msg in enumerate(messages):
        content = _clean_history_content(msg.get("content", ""))
        role = msg.get("role", "user")

        if i >= full_cutoff:
            # Tier 1: Last 6 messages — keep full
            compressed.append({"role": role, "content": content})
        elif i >= middle_cutoff:
            # Tier 2: Middle — user questions full, assistant trimmed
            if role == "assistant" and len(content) > 300:
                content = content[:300].rsplit(". ", 1)[0] + ". [...]"
            compressed.append({"role": role, "content": content})
        else:
            # Tier 3: Oldest — only keep user questions as topic markers
            if role == "user":
                compressed.append({"role": role, "content": content[:200]})

    return compressed


def _clean_history_content(content: str) -> str:
    """Strip old RAG context blocks from a history message."""
    if not content:
        return content

    # Strip KNOWLEDGE BASE CONTEXT blocks
    if "KNOWLEDGE BASE CONTEXT:" in content:
        parts = content.split("QUESTION:")
        if len(parts) > 1:
            return parts[-1].strip()
        lines = content.split("\n")
        return " ".join(
            l for l in lines
            if not l.startswith("=") and not l.startswith("-" * 10)
            and not l.startswith("[Source") and "KNOWLEDGE BASE" not in l
        )[:400]

    # Strip Reference information blocks (from local model format)
    if "Reference information:" in content:
        parts = content.split("Based on the reference information above, answer this question:")
        if len(parts) > 1:
            return parts[-1].strip()

    return content


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
    llm_model = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
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
        cloud_user_message = f"""متن پایگاه دانش:
{context}

سوال: {question}

لطفاً یک پاسخ دقیق و فنی بر اساس متن پایگاه دانش بالا ارائه دهید. منابع را با شماره [منبع N] ارجاع دهید.

IMPORTANT: You MUST answer entirely in Persian/Farsi. Do NOT answer in English."""
    else:
        cloud_user_message = f"""KNOWLEDGE BASE CONTEXT:
{context}

QUESTION: {question}

Provide a precise, technical answer based strictly on the knowledge base context above. Reference sources by their [Source N] numbers."""

    # Smart history compression: keep recent turns full, summarize older ones
    history = _compress_history((conversation_history or [])[-10:])

    # ── Farsi: skip local model entirely ──────────────────────────────────────
    # The local 8B model was fine-tuned on English Q&A only. It cannot generate
    # Farsi text. Route Farsi queries directly to GPT-4o-mini (scored 4.70/5).
    local_failed = False
    if language == "fa":
        logger.info("Farsi query detected — routing directly to GPT-4o-mini (local model is English-only)")
        _log_fallback_event("farsi_query", question, language)
        local_failed = True
    else:
        # ── Try local LLM (English only) ──────────────────────────────────────
        # CRITICAL: Strip RAG formatting to match training data format.
        # The model was trained on plain Q&A with the standard Glass Expert AI prompt.
        # It has never seen [Source N], Type:, Language:, Relevance: headers.
        plain_context = _strip_rag_formatting(context)

        # Truncate for 8B model token budget (~8192 tokens total)
        # System prompt ~200 tokens + user message ~3000 tokens + response ~1500 tokens
        local_context_limit = 12000  # chars (~3000 tokens) — more context for bge-m3 chunks
        if len(plain_context) > local_context_limit:
            plain_context = plain_context[:local_context_limit]
            logger.debug(f"Truncated context for local model: {len(context)} → {local_context_limit} chars")

        # User message: minimal wrapper — the system prompt already has all rules.
        # The v2 model was trained on plain Q&A. Keep user message clean and simple.
        local_user_message = f"""Reference information:
{plain_context}

Based on the reference information above, answer this question: {question}"""

        local_url = llm_url if llm_url else "http://localhost:8000/v1"
        try:
            # Local model: use compressed history (8 messages = 4 turns for continuity)
            local_history = history[-8:] if history else None
            # Local model gets more tokens for detailed technical answers
            local_max_tokens = max(max_tokens, 1500)
            answer = await _call_openai_compatible(
                base_url=local_url,
                api_key=llm_api_key,
                model=llm_model,
                system_prompt=LOCAL_SYSTEM_PROMPT,
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
            else:
                return answer, llm_model
        except Exception as e:
            logger.warning(f"Local LLM at {local_url} not available: {e}")
            _log_fallback_event("local_unavailable", question, language, local_error=str(e))
            local_failed = True

    # ── Fallback: OpenAI GPT (Farsi queries or local model failure) ───────────
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

    # Build extra params for local vLLM — tuned to prevent degenerate output
    extra = {}
    if "localhost" in base_url or "127.0.0.1" in base_url:
        extra["extra_body"] = {
            "repetition_penalty": 1.12,  # prevent repetitive outputs without killing technical terms
            "top_p": 0.90,              # slightly tighter nucleus sampling
            "top_k": 40,               # focused vocabulary
        }

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=30.0,
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
