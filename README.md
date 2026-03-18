# Glass Expert AI — Phase 3: Production RAG API

Bilingual (English + Farsi) RAG-powered glass science assistant for engineers and scientists.
Built with FastAPI, pgvector, bge-m3, bge-reranker-v2-m3, and a fine-tuned LLM.

---

## Architecture
```
React UI (:8080)
  → FastAPI API v3.1.0
  → JWT Auth (register/login/me)
  → Rate Limiting (30 req/min)
  → RAG Pipeline:
      1. bge-m3 query embedding (sentence-transformers + instruction prefix)
      2. Dense ANN search (pgvector HNSW) on documents_bgem3    → top_k × 4 candidates
      3. Sparse BM25 search (bge-m3 tokens via tsvector)        → top_k × 2 candidates
      4. Reciprocal Rank Fusion (RRF, k=60)                     → merged ranked list
      5. bge-reranker-v2-m3 cross-encoder                       → top_k final results
  → Redis cache (:6379)
  → Fine-tuned LLM (:8000) / OpenAI fallback
  → PostgreSQL + pgvector (:5432)
```

---

## Knowledge Base — documents_bgem3 table

| Source Type | Chunks   |
|-------------|----------|
| manual      | 143,140  |
| paper       | 60,749   |
| qa_pair     | 32,793   |
| textbook    | 10,309   |
| sop         | 10       |
| **Total**   | **247,001** |

Languages: English + Farsi
Embedding model: BAAI/bge-m3 (1024-dim, multilingual, sentence-transformers with instruction prefix)
Vector index: HNSW (m=16, ef_construction=64)
Reranker top score: 0.9980

---

## Quick Start

### Step 1 — Start Docker services
```powershell
docker-compose -f docker/docker-compose.yml up -d
```

### Step 2 — Activate virtual environment
```powershell
.\venv\Scripts\activate
```

### Step 3 — Install dependencies
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install FlagEmbedding==1.3.3
```

### Step 4 — Configure environment
```powershell
copy .env.example .env
# Edit .env — set JWT_SECRET_KEY, LLM_BASE_URL
```

### Step 5 — Start LLM server (requires model weights from Engineer Z)
```powershell
python model_service/serve.py
```

### Step 6 — Start the API server
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Step 7 — Open the app
- Chat UI: http://localhost:8080/
- API Docs: http://localhost:8080/docs
- Health: http://localhost:8080/api/v1/health

---

## API Endpoints

All endpoints except health require JWT authentication via `Authorization: Bearer <token>`.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | /api/v1/health | Public | System health check |
| POST | /api/v1/auth/register | Public | Create account |
| POST | /api/v1/auth/login | Public | Login (returns JWT) |
| GET | /api/v1/auth/me | Required | Current user profile |
| POST | /api/v1/query | Required | RAG Q&A — main endpoint |
| POST | /api/v1/analyze | Required | Glass composition analysis |
| POST | /api/v1/design | Required | Glass composition design |
| POST | /api/v1/troubleshoot | Required | Defect diagnosis |
| GET | /api/v1/conversations | Required | List conversations |
| GET | /api/v1/conversations/{id} | Required | Get conversation messages |
| POST | /api/v1/conversations | Required | Create conversation |
| PATCH | /api/v1/conversations/{id} | Required | Rename conversation |
| DELETE | /api/v1/conversations/{id} | Required | Delete conversation |
| POST | /api/v1/feedback | Required | Submit answer feedback |
| POST | /api/v1/ingest | Admin | Document ingestion |

---

## LLM Configuration

Tried in order:
1. **Local model server** (`model_service/serve.py`) — fine-tuned Llama-3.1-8B on port 8000
2. **OpenAI GPT-4o-mini** — fallback if local model unavailable
3. **Retrieval-only** — returns raw retrieved context if no LLM available

To start the local model server (requires weights from Engineer Z):
```powershell
python model_service/serve.py
```

---

## Evaluation

Golden evaluation set (50 questions) available at `data/evaluation/golden_eval_set.jsonl`.

Run evaluation after LLM is connected:
```powershell
python scripts/evaluation/run_golden_eval.py --url http://localhost:8080 --label arjun-247k
```

Target: beat Engineer Z's baseline of **4.08/5.00 judge score, 66.58% composite**.

---

## Services

| Service | Port | Credentials |
|---------|------|-------------|
| API Server | 8080 | JWT auth |
| LLM Server | 8000 | token-glass-ai |
| PostgreSQL | 5432 | glassai / glassai_secret |
| Redis | 6379 | — |
| pgAdmin | 5050 | admin@glassai.com / admin |

---

## Project Structure
```
glass-expert-ai/
├── api/
│   ├── auth.py                ← JWT auth (register/login/me) + UserInToken
│   ├── database.py            ← ThreadedConnectionPool (min=2, max=20)
│   ├── main.py                ← FastAPI v3.1.0 — CORS, rate limiting, DB pool init
│   ├── models/schemas.py      ← Pydantic request/response models
│   └── routers/               ← query, analyze, design, troubleshoot,
│                                 feedback, conversations, health, ingest
├── ingestion/
│   ├── embedder.py            ← bge-m3 via sentence-transformers + instruction prefix
│   ├── ingest_jsonl.py        ← Q&A pair ingestion — bge-m3 embeddings
│   └── ingest.py              ← Main ingestion pipeline
├── retrieval/
│   ├── retriever.py           ← Hybrid: dense + sparse BM25 + RRF → documents_bgem3
│   ├── reranker.py            ← bge-reranker-v2-m3 cross-encoder
│   └── llm.py                 ← LLM integration + fallback governance
├── model_service/
│   ├── serve.py               ← Local LLM server (HuggingFace + LoRA)
│   └── serve_vllm.py          ← vLLM server (Linux only)
├── data/evaluation/
│   ├── golden_eval_set.jsonl  ← 50-question English evaluation set
│   └── persian_eval_set.jsonl ← Farsi evaluation set
├── scripts/evaluation/
│   └── run_golden_eval.py     ← Evaluation runner with LLM-as-judge scoring
├── docker/
│   ├── docker-compose.yml     ← PostgreSQL + Redis + pgAdmin
│   └── init.sql               ← Database schema (HNSW index)
├── .env.example               ← Environment template
└── README.md
```

---

## Branches

| Branch | Owner | Description |
|--------|-------|-------------|
| `main` | MTAAI | Default branch |
| `Arjun` | Arjun | Application + RAG pipeline (this branch) |
| `glass-expert-ai` | Engineer Z | LLM fine-tuning + evaluation |