"""
Glass Expert AI — LLM Integration
Supports:
  - Local vLLM (Llama-3-8B) via OpenAI-compatible API
  - OpenAI GPT-4 (fallback or cloud mode)
  - Retrieval-only mode (no LLM, returns formatted context)
"""
import os
from loguru import logger

# Singleton client cache — avoids creating a new connection per request
_client_cache: dict = {}

# System prompts for expert glass science answers
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
2. Follow with detailed technical explanation using numbered points.
3. Include specific compositions (e.g., 72% SiO2, 14% Na2O) and property values FROM the context.
4. For processes, list ALL stages with their specific temperature ranges and conditions.
5. Reference named scientists, equations (e.g., Abbe number V=(n_d-1)/(n_F-n_C)), and standards (ISO, ASTM) when they appear in the context.
6. Use precise technical terms: network formers, network modifiers, bridging oxygen (BO), non-bridging oxygen (NBO), coordination number, fining agents, devitrification, etc.
7. Cite sources as [Source 1], [Source 2], etc. for every key fact.
8. End with a brief summary if the answer covers multiple aspects.
9. Keep answers focused, quantitative, and specific — never give vague generalizations."""

SYSTEM_PROMPT_FA = """شما Glass Expert AI هستید، یک دستیار تخصصی با تخصص سطح دکترا برای دانشمندان شیشه و مهندسان تولید.

قواعد حیاتی:
- فقط بر اساس متن پایگاه دانش ارائه شده پاسخ دهید — هرگز اطلاعات جعلی ارائه ندهید
- مقادیر، فرمول‌ها و ترکیبات دقیق را مستقیماً از متن استخراج و نقل کنید
- اگر متن اطلاعات کافی ندارد، بگویید «متن ارائه شده این اطلاعات را مشخص نمی‌کند» — حدس نزنید

قالب‌بندی:
1. با تعریف یا پاسخ مستقیم شروع کنید
2. از لیست شماره‌دار برای فرآیندها، قوانین و روش‌ها استفاده کنید
3. ترکیبات خاص (مثلاً ۷۲٪ SiO2) و مقادیر خواص را از متن ذکر کنید
4. نام سند منبع را در کروشه ذکر کنید، مثلاً [منبع ۱]
5. اگر منابع مختلف اطلاعات متناقضی دارند، تناقض را ذکر کنید
6. مقادیر عددی را با واحد و شرایط (دما، فشار، ترکیب) ذکر کنید"""


def _is_degenerate(text: str) -> bool:
    """Detect repetitive/nonsensical LLM output that should trigger fallback."""
    if not text or len(text.split()) < 10:
        return True
    # Check for excessive repetition: if any 4-word phrase repeats 5+ times
    words = text.split()
    if len(words) > 20:
        phrases = [" ".join(words[i:i+4]) for i in range(len(words) - 3)]
        from collections import Counter
        most_common = Counter(phrases).most_common(1)
        if most_common and most_common[0][1] >= 5:
            return True
    return False


async def generate_answer(
    question: str,
    context: str,
    language: str = "en",
    conversation_history: list | None = None,
) -> tuple[str, str]:
    """
    Generate an expert answer using the configured LLM.

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

    system_prompt = SYSTEM_PROMPT_FA if language == "fa" else SYSTEM_PROMPT_EN

    # Add conversation continuity instruction when history is present
    if conversation_history:
        system_prompt += (
            "\n\nYou are in an ongoing conversation. Prior messages are provided for context. "
            "Use them to understand follow-up questions and maintain continuity."
        )

    if language == "fa":
        user_message = f"""متن پایگاه دانش:
{context}

سوال: {question}

لطفاً یک پاسخ دقیق و فنی بر اساس متن پایگاه دانش بالا ارائه دهید. منابع را با شماره [منبع N] ارجاع دهید.

IMPORTANT: You MUST answer entirely in Persian/Farsi. Do NOT answer in English."""
    else:
        user_message = f"""KNOWLEDGE BASE CONTEXT:
{context}

QUESTION: {question}

Provide a precise, technical answer based strictly on the knowledge base context above. Reference sources by their [Source N] numbers."""

    # Cap history at 10 messages (5 turns)
    history = (conversation_history or [])[-10:]

    # ── Try local LLM (supports both English and Farsi via Llama 3.1 base) ─────
    local_url = llm_url if llm_url else "http://localhost:8000/v1"
    local_failed = False
    try:
        answer = await _call_openai_compatible(
            base_url=local_url,
            api_key=llm_api_key,
            model=llm_model,
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
            conversation_history=history,
        )
        # Quality gate: detect degenerate output (repetitive/nonsensical)
        if _is_degenerate(answer):
            logger.warning(f"Local model produced degenerate output ({len(answer)} chars), falling back")
            local_failed = True
        else:
            return answer, llm_model
    except Exception as e:
        logger.warning(f"Local LLM at {local_url} not available: {e}")
        local_failed = True

    # ── Fallback: OpenAI GPT (only when local model fails or degenerates) ─────
    if local_failed:
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if openai_key:
            try:
                logger.info("Using OpenAI fallback due to local model failure")
                answer = await _call_openai_compatible(
                    base_url="https://api.openai.com/v1",
                    api_key=openai_key,
                    model="gpt-4o-mini",
                    system_prompt=system_prompt,
                    user_message=user_message,
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

    # Build extra params — repetition_penalty for local vLLM to prevent degenerate output
    extra = {}
    if "localhost" in base_url or "127.0.0.1" in base_url:
        extra["extra_body"] = {"repetition_penalty": 1.15}

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **extra,
    )

    return response.choices[0].message.content.strip()
