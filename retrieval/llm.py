"""
Glass Expert AI — LLM Integration
Supports:
  - Local vLLM (Llama-3-8B) via OpenAI-compatible API
  - OpenAI GPT-4 (fallback or cloud mode)
  - Retrieval-only mode (no LLM, returns formatted context)
"""
import os
from loguru import logger

# System prompts for expert glass science answers
SYSTEM_PROMPT_EN = """You are Glass Expert AI, a highly specialized assistant for glass scientists and manufacturing engineers.

Your knowledge base contains textbooks, research papers, SOPs, and standards on glass science and technology.

Instructions:
- Answer ONLY based on the provided knowledge base context
- Be precise and technical — your audience are professional engineers
- Include specific values, formulas, and parameters when available
- If the context does not contain enough information, say so clearly
- Do NOT make up information not present in the context
- Format your answer clearly with sections if the answer is long
- Cite the source document name when referencing specific data"""

SYSTEM_PROMPT_FA = """شما Glass Expert AI هستید، یک دستیار تخصصی برای دانشمندان شیشه و مهندسان تولید.

پایگاه دانش شما شامل کتب درسی، مقالات تحقیقاتی، دستورالعمل‌های عملیاتی و استانداردهای علم و فناوری شیشه است.

دستورالعمل‌ها:
- فقط بر اساس متن پایگاه دانش ارائه شده پاسخ دهید
- دقیق و فنی باشید — مخاطبان شما مهندسان حرفه‌ای هستند
- در صورت وجود، مقادیر، فرمول‌ها و پارامترهای خاص را ذکر کنید
- اگر متن اطلاعات کافی ندارد، صریحاً بگویید
- اطلاعاتی که در متن نیست را اختراع نکنید"""


async def generate_answer(
    question: str,
    context: str,
    language: str = "en",
) -> tuple[str, str]:
    """
    Generate an expert answer using the configured LLM.

    Returns:
        Tuple of (answer_text, model_name_used)
    """
    llm_url = os.getenv("LLM_BASE_URL", "")
    llm_model = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    llm_api_key = os.getenv("LLM_API_KEY", "token-glass-ai")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    system_prompt = SYSTEM_PROMPT_FA if language == "fa" else SYSTEM_PROMPT_EN

    user_message = f"""KNOWLEDGE BASE CONTEXT:
{context}

QUESTION: {question}

Please provide a precise, technical answer based on the knowledge base context above."""

    # ── Try local vLLM first ───────────────────────────────────────────────────
    if llm_url and llm_url != "http://localhost:8000/v1":
        try:
            answer = await _call_openai_compatible(
                base_url=llm_url,
                api_key=llm_api_key,
                model=llm_model,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return answer, llm_model
        except Exception as e:
            logger.warning(f"Local vLLM failed: {e}. Trying OpenAI fallback...")

    # ── Try local vLLM on default port ────────────────────────────────────────
    try:
        answer = await _call_openai_compatible(
            base_url="http://localhost:8000/v1",
            api_key=llm_api_key,
            model=llm_model,
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return answer, llm_model
    except Exception as e:
        logger.warning(f"Local vLLM on port 8000 not available: {e}")

    # ── Fallback: OpenAI GPT ───────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            answer = await _call_openai_compatible(
                base_url="https://api.openai.com/v1",
                api_key=openai_key,
                model="gpt-4o-mini",
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return answer, "gpt-4o-mini"
        except Exception as e:
            logger.warning(f"OpenAI fallback failed: {e}")

    # ── Final fallback: return context directly ────────────────────────────────
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
) -> str:
    """Call any OpenAI-compatible API (vLLM, OpenAI, etc.)."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()
