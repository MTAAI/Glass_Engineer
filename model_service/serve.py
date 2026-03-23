"""
Glass Expert AI — Model Server v3.0
====================================
Tiered architecture: RAG context + Fine-tuned LLM + OpenAI fallback.

API Endpoints:
  - GET  /v1/models
  - POST /v1/chat/completions   (OpenAI-compatible, works with RAG context)
  - POST /query                 (RAG-aware query with fallback)
  - GET  /health

Environment variables:
  BASE_MODEL_ID   - HuggingFace model ID (default: unsloth/Qwen2.5-14B-Instruct-bnb-4bit)
  LORA_ADAPTER    - Path to LoRA adapter (default: models/qwen14b-glass-expert/adapter)
  MODEL_NAME      - Model name for API responses (default: glass-expert)
  PORT            - Server port (default: 8000)
  MAX_NEW_TOKENS  - Max generation tokens (default: 1024)
  DEFAULT_TEMP    - Default temperature (default: 0.1)
  OPENAI_API_KEY  - OpenAI API key for fallback (optional)
  OPENAI_MODEL    - Fallback model (default: gpt-4.1-mini)

Usage:
    python model_service/serve.py
    LORA_ADAPTER=models/glass-expert-v3/final MODEL_NAME=glass-expert-v3 python model_service/serve.py
"""

import os, time, uuid, asyncio, logging, threading, json
from typing import Optional, List
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from peft import PeftModel

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL_ID  = os.environ.get("BASE_MODEL_ID", "/root/glass-training/models/Qwen2.5-14B-Instruct-bnb-4bit")
LORA_ADAPTER   = os.environ.get("LORA_ADAPTER", os.path.join(PROJECT_ROOT, "models", "qwen14b-glass-expert", "adapter"))
MERGED_MODEL   = os.environ.get("MERGED_MODEL", os.path.join(PROJECT_ROOT, "models", "qwen14b-glass-expert-merged"))
USE_MERGED     = os.environ.get("USE_MERGED", "false").lower() in ("true", "1", "yes")
MODEL_NAME     = os.environ.get("MODEL_NAME", "glass-expert-qwen14b")
PORT           = int(os.environ.get("PORT", "8000"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "1024"))
DEFAULT_TEMP   = float(os.environ.get("DEFAULT_TEMP", "0.1"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
QUANTIZE_4BIT  = os.environ.get("QUANTIZE_4BIT", "true").lower() in ("true", "1", "yes")

SYSTEM_PROMPT = """You are Glass Expert AI, a highly specialized assistant for glass scientists and manufacturing engineers with PhD-level expertise.

When answering questions:
1. Always enumerate specific numbered items when listing processes, rules, causes, or methods.
2. Include exact numerical values, ranges, and units. Always use degrees Celsius for temperatures, not Kelvin.
3. Name specific glass types (e.g., soda-lime-silica, borosilicate 3.3, E-glass), their exact compositions (e.g., 72% SiO2, 14% Na2O), and precise property values.
4. Reference named scientists (e.g., Zachariasen), equations (e.g., Abbe number V=(n_d-1)/(n_F-n_C)), and standards (ISO, ASTM, EN) when relevant.
5. Use precise technical terms: network formers, network modifiers, bridging oxygen (BO), non-bridging oxygen (NBO), coordination number, fining agents, devitrification, etc.
6. Structure answers clearly: brief definition, then numbered key points with specific details.
7. Never give vague generalizations — always be specific, quantitative, and factual."""

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────────
model = None
tokenizer = None
openai_client = None

# ── Pydantic models ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[Message]
    temperature: Optional[float] = DEFAULT_TEMP
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    stream: Optional[bool] = False

class QueryRequest(BaseModel):
    question: str
    context: Optional[List[str]] = Field(default=None, description="RAG-retrieved context passages")
    use_fallback: Optional[bool] = Field(default=True, description="Fall back to OpenAI if local model fails")
    max_tokens: Optional[int] = MAX_NEW_TOKENS
    temperature: Optional[float] = DEFAULT_TEMP

class Source(BaseModel):
    text: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    model_used: str
    sources: Optional[List[str]] = None
    latency_ms: float

# ── FastAPI app ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, openai_client
    logger.info(f"Loading model: {MODEL_NAME}")

    # Prefer merged model (no PEFT overhead, ~30-40% faster inference)
    if USE_MERGED and os.path.isdir(MERGED_MODEL):
        logger.info(f"  Loading pre-merged model from: {MERGED_MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL, device_map="auto", torch_dtype=torch.bfloat16
        )
        model.eval()
        logger.info("Merged model loaded and ready! (no PEFT overhead)")
    else:
        logger.info(f"  Base: {BASE_MODEL_ID}")
        logger.info(f"  LoRA: {LORA_ADAPTER}")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if QUANTIZE_4BIT:
            logger.info("Loading with 4-bit quantization (NF4)")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID, device_map="auto", quantization_config=bnb_config
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID, device_map="auto", torch_dtype=torch.bfloat16
            )
        model = PeftModel.from_pretrained(base, LORA_ADAPTER)
        model.eval()
        logger.info(f"LoRA model loaded and ready! (4-bit: {QUANTIZE_4BIT})")

    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            openai_client = OpenAI(api_key=OPENAI_API_KEY)
            logger.info(f"OpenAI fallback enabled (model: {OPENAI_MODEL})")
        except ImportError:
            logger.warning("openai package not installed — fallback disabled")
    else:
        logger.info("No OPENAI_API_KEY set — fallback disabled")

    yield
    logger.info("Shutting down server.")

app = FastAPI(
    title="Glass Expert AI Server",
    version="3.0.0",
    description="Tiered: RAG context + Fine-tuned LLM + OpenAI fallback",
    lifespan=lifespan,
)

_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Local model inference ─────────────────────────────────────────────────────
def _build_prompt(messages: List[Message]) -> str:
    # Respect the client's system prompt (e.g., RAG grounding instructions from llm.py).
    # Only inject the default SYSTEM_PROMPT when no system prompt is provided.
    has_system = any(m.role == "system" for m in messages)
    if not has_system:
        messages.insert(0, Message(role="system", content=SYSTEM_PROMPT))

    return tokenizer.apply_chat_template(
        [{"role": m.role, "content": m.content} for m in messages],
        tokenize=False, add_generation_prompt=True,
    )

def _generate_sync(prompt: str, temperature: float, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=max(temperature, 1e-6), do_sample=temperature > 0.01,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

async def _generate_stream(prompt, temperature, max_new_tokens, completion_id):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)
    thread = threading.Thread(target=model.generate, kwargs=dict(
        **inputs, max_new_tokens=max_new_tokens,
        temperature=max(temperature, 1e-6), do_sample=temperature > 0.01,
        pad_token_id=tokenizer.eos_token_id, streamer=streamer,
    ))
    thread.start()
    for chunk in streamer:
        if chunk:
            data = {"id": completion_id, "object": "chat.completion.chunk",
                    "created": int(time.time()), "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": chunk},
                                 "finish_reason": None}]}
            yield f"data: {json.dumps(data)}\n\n"
        await asyncio.sleep(0)
    thread.join()
    yield "data: [DONE]\n\n"

# ── OpenAI fallback ───────────────────────────────────────────────────────────
def _openai_fallback(question: str, context: Optional[List[str]] = None, max_tokens: int = 1024, temperature: float = 0.1) -> str:
    if not openai_client:
        raise RuntimeError("OpenAI fallback not configured")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    user_content = question
    if context:
        ctx_text = "\n\n".join(f"[Source {i+1}]: {c}" for i, c in enumerate(context))
        user_content = f"Use the following reference material to answer:\n\n{ctx_text}\n\nQuestion: {question}"
    messages.append({"role": "user", "content": user_content})

    resp = openai_client.chat.completions.create(
        model=OPENAI_MODEL, messages=messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    return resp.choices[0].message.content.strip()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/v1/models")
async def list_models():
    models = [{"id": MODEL_NAME, "object": "model", "created": 1700000000, "owned_by": "glass-expert-ai"}]
    if openai_client:
        models.append({"id": f"{OPENAI_MODEL} (fallback)", "object": "model", "created": 1700000000, "owned_by": "openai"})
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    prompt = _build_prompt(request.messages)
    temp = request.temperature or DEFAULT_TEMP
    max_tok = request.max_tokens or MAX_NEW_TOKENS
    if request.stream:
        return StreamingResponse(
            _generate_stream(prompt, temp, max_tok, completion_id),
            media_type="text/event-stream",
        )
    answer = _generate_sync(prompt, temp, max_tok)
    return {
        "id": completion_id, "object": "chat.completion",
        "created": int(time.time()), "model": MODEL_NAME,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": answer},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": -1, "completion_tokens": -1, "total_tokens": -1},
    }

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    RAG-aware query endpoint with OpenAI fallback.

    Flow:
    1. If RAG context is provided, inject it into the prompt
    2. Try the fine-tuned local model first
    3. If local model fails and use_fallback=True, try OpenAI
    """
    start = time.time()
    question = request.question
    context = request.context
    temp = request.temperature or DEFAULT_TEMP
    max_tok = request.max_tokens or MAX_NEW_TOKENS

    # Build user message with RAG context
    if context:
        ctx_text = "\n\n".join(f"[Source {i+1}]: {c}" for i, c in enumerate(context))
        user_content = f"Use the following reference material to answer accurately:\n\n{ctx_text}\n\nQuestion: {question}"
    else:
        user_content = question

    # Tier 1: Try fine-tuned local model
    try:
        if model is None:
            raise RuntimeError("Model not loaded")
        messages = [Message(role="system", content=SYSTEM_PROMPT),
                    Message(role="user", content=user_content)]
        prompt = _build_prompt(messages)
        answer = _generate_sync(prompt, temp, max_tok)

        if answer and len(answer.split()) >= 5:
            return QueryResponse(
                answer=answer,
                model_used=MODEL_NAME,
                sources=[c[:200] + "..." if len(c) > 200 else c for c in context] if context else None,
                latency_ms=round((time.time() - start) * 1000, 0),
            )
        logger.warning(f"Local model gave short answer ({len(answer.split())} words), trying fallback")
    except Exception as e:
        logger.error(f"Local model failed: {e}")

    # Tier 2: OpenAI fallback
    if request.use_fallback and openai_client:
        try:
            answer = _openai_fallback(question, context, max_tok, temp)
            return QueryResponse(
                answer=answer,
                model_used=f"{OPENAI_MODEL} (fallback)",
                sources=[c[:200] + "..." if len(c) > 200 else c for c in context] if context else None,
                latency_ms=round((time.time() - start) * 1000, 0),
            )
        except Exception as e:
            logger.error(f"OpenAI fallback failed: {e}")
            raise HTTPException(status_code=500, detail=f"All models failed: {e}")

    raise HTTPException(status_code=500, detail="Local model failed and no fallback configured")

@app.get("/health")
async def health():
    return {
        "status": "ready" if model is not None else "loading",
        "model": MODEL_NAME,
        "lora_adapter": LORA_ADAPTER,
        "fallback": OPENAI_MODEL if openai_client else None,
    }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Glass Expert AI Server v3.0 on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
