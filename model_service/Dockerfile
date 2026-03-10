# ============================================================
# Glass Expert AI — LLM Server Dockerfile (vLLM)
# High-performance inference with PagedAttention
# Requires NVIDIA GPU runtime (--gpus all)
#
# Usage:
#   docker build -t glass-expert-llm -f model_service/Dockerfile .
#   docker run --gpus all -p 8000:8000 \
#     -v /path/to/models:/app/models \
#     -e LORA_ADAPTER=/app/models/glass-expert-v2/final \
#     glass-expert-llm
# ============================================================

FROM vllm/vllm-openai:latest

WORKDIR /app

# Copy server scripts (vLLM for production, HF for fallback)
COPY model_service/serve_vllm.py ./model_service/serve_vllm.py
COPY model_service/serve.py ./model_service/serve.py

# Model weights are mounted at runtime, not baked into image
# Mount to /app/models via -v flag

# Env defaults
ENV PORT=8000
ENV MODEL_NAME=glass-expert
ENV BASE_MODEL_ID=meta-llama/Meta-Llama-3.1-8B-Instruct
ENV LORA_ADAPTER=/app/models/glass-expert-v2/final
ENV MAX_MODEL_LEN=4096
ENV GPU_MEM_UTIL=0.85
ENV LLM_API_KEY=token-glass-ai
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "model_service/serve_vllm.py"]
