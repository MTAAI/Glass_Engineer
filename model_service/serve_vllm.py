"""
Glass Expert AI — vLLM Model Server
=====================================
High-performance inference using vLLM with PagedAttention.
3-5x faster than raw HuggingFace, same OpenAI-compatible API.

Usage (standalone):
    python model_service/serve_vllm.py

Usage (CLI):
    vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
        --enable-lora --lora-modules glass-expert=models/glass-expert-v2/final \
        --port 8000 --dtype bfloat16 --max-model-len 4096 \
        --gpu-memory-utilization 0.85

Environment variables:
    BASE_MODEL_ID   - HuggingFace model ID (default: meta-llama/Meta-Llama-3.1-8B-Instruct)
    LORA_ADAPTER    - Path to LoRA adapter (default: models/glass-expert-v2/final)
    MODEL_NAME      - Model name for API (default: glass-expert)
    PORT            - Server port (default: 8000)
    MAX_MODEL_LEN   - Max sequence length (default: 4096)
    GPU_MEM_UTIL    - GPU memory utilization 0-1 (default: 0.85)

Note: vLLM requires Linux. For Windows local dev, use serve.py instead.
"""
import os
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
LORA_ADAPTER = os.environ.get("LORA_ADAPTER", os.path.join(PROJECT_ROOT, "models", "glass-expert-v2", "final"))
MODEL_NAME = os.environ.get("MODEL_NAME", "glass-expert")
PORT = os.environ.get("PORT", "8000")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "4096")
GPU_MEM_UTIL = os.environ.get("GPU_MEM_UTIL", "0.85")
API_KEY = os.environ.get("LLM_API_KEY", "token-glass-ai")

def main():
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", BASE_MODEL_ID,
        "--enable-lora",
        "--lora-modules", f"{MODEL_NAME}={LORA_ADAPTER}",
        "--port", PORT,
        "--dtype", "bfloat16",
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--api-key", API_KEY,
        "--served-model-name", MODEL_NAME,
        "--trust-remote-code",
    ]

    print(f"Starting vLLM server on port {PORT}")
    print(f"  Base model: {BASE_MODEL_ID}")
    print(f"  LoRA adapter: {LORA_ADAPTER}")
    print(f"  Model name: {MODEL_NAME}")
    print(f"  Max seq len: {MAX_MODEL_LEN}")
    print(f"  GPU memory: {GPU_MEM_UTIL}")
    print(f"  Command: {' '.join(cmd)}")

    subprocess.run(cmd)


if __name__ == "__main__":
    main()
