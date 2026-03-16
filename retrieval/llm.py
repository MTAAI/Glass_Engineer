"""
Glass Expert AI — LLM Integration
Supports:
  - Local vLLM (Llama-3-8B) via OpenAI-compatible API
  - OpenAI GPT-4 (fallback or cloud mode)
  - Retrieval-only mode (no LLM, returns formatted context)
"""
import os
import re
from loguru import logger

# Singleton client cache — avoids creating a new connection per request
_client_cache: dict = {}

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
LOCAL_SYSTEM_PROMPT = """You are Glass Expert AI, a highly specialized assistant trained on thousands of glass science research papers, textbooks, and material databases. You provide accurate, detailed, and technically precise answers about glass composition, properties, manufacturing processes, defects, characterization, and applications. Always cite relevant glass science principles in your answers."""


def _strip_rag_formatting(context: str) -> str:
    """
    Strip RAG metadata formatting from context to produce plain text.

    The local model was trained on plain Q&A — it has never seen [Source N],
    Type:, Language:, Relevance: metadata headers. These confuse the 8B model
    into producing meta-commentary instead of answers.

    Input format (from format_context_for_llm):
        KNOWLEDGE BASE CONTEXT:
        ==================================================
        [Source 1] Title (Type: paper | Language: EN | Relevance: 83%)
        actual content here...
        ----------------------------------------
        [Source 2] Another Title (Type: textbook | Language: EN | Relevance: 75%)
        more content...
        ----------------------------------------

    Output format (plain text for local model):
        actual content here...

        more content...
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
        # Skip [Source N] metadata lines
        if re.match(r'^\[Source \d+\]', stripped):
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
    meta_patterns = [
        "the text is written",
        "the text is a reference",
        "this text describes",
        "the passage discusses",
        "the provided text",
        "the above text",
        "this response is referenced",
    ]
    if any(p in lower for p in meta_patterns):
        return True

    # Check for truncated/fragmented output (no complete sentence)
    if len(text) < 80 and "." not in text and ":" not in text and "،" not in text:
        return True

    return False


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
        local_failed = True
    else:
        # ── Try local LLM (English only) ──────────────────────────────────────
        # CRITICAL: Strip RAG formatting to match training data format.
        # The model was trained on plain Q&A with the standard Glass Expert AI prompt.
        # It has never seen [Source N], Type:, Language:, Relevance: headers.
        plain_context = _strip_rag_formatting(context)

        # Truncate for 8B model token budget (~8192 tokens total)
        # System prompt ~100 tokens + user message ~2500 tokens + response ~1500 tokens
        local_context_limit = 6000  # chars (~1500 tokens) — leave room for response
        if len(plain_context) > local_context_limit:
            plain_context = plain_context[:local_context_limit]
            logger.debug(f"Truncated context for local model: {len(context)} → {local_context_limit} chars")

        # User message: context first, then question with extraction instruction.
        # The explicit "Based on the reference" nudges the model to use the context
        # rather than hallucinating from its parametric memory.
        local_user_message = f"""Reference information:
{plain_context}

Based on the reference information above, answer this question: {question}

Important: Use specific numbers, temperatures, and compositions from the reference information. Do not make up values."""

        local_url = llm_url if llm_url else "http://localhost:8000/v1"
        try:
            # Local model: use shorter history (4 messages = 2 turns) to save tokens
            local_history = history[-4:] if history else None
            # Local model gets more tokens to complete its answer
            local_max_tokens = max(max_tokens, 1200)
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
                logger.warning(f"Falling back to OpenAI")
                local_failed = True
            else:
                return answer, llm_model
        except Exception as e:
            logger.warning(f"Local LLM at {local_url} not available: {e}")
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
    """Call any OpenAI-compatible API (vLLM, OpenAI, etc.)."""
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
            "repetition_penalty": 1.25,  # stronger penalty to prevent loops
            "top_p": 0.9,               # nucleus sampling for diversity
            "top_k": 40,                # limit vocabulary to top 40 tokens
        }

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
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
