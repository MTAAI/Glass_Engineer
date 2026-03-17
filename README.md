# Glass Expert AI — Phase 3: Production RAG API

Bilingual (English + Farsi) RAG-powered glass science assistant for engineers and scientists.
Built with FastAPI, pgvector, bge-m3, bge-reranker-v2-m3, and a fine-tuned LLM.

---

## Architecture
```
React UI (:8080)
  → FastAPI API
  → JWT Auth (register/login/me)
  → RAG Pipeline:
      1. bge-m3 query embedding (sentence-transformers + instruction prefix)
      2. Dense ANN search (pgvector HNSW)       → top_k × 4 candidates
      3. Sparse BM25 search (bge-m3 tokens)     → top_k × 2 candidates
      4. Reciprocal Rank Fusion (RRF, k=60)     → merged ranked list
      5. bge-reranker-v2-m3 cross-encoder       → precise top_k results
  → Redis cache (:6379)
  → Fine-tuned LLM (:8000) / OpenAI fallback
  → PostgreSQL + pgvector (:5432)
```

---

## Knowledge Base

| Source Type | Chunks |
|-------------|--------|
| manual      | 143,140 |
| paper       | 60,749 |
| qa_pair     | 32,793 |
| textbook    | 10,309 |
| sop         | 10 |
| **Total**   | **247,001** |

Languages: English + Farsi  
Embedding model: BAAI/bge-m3 (1024-dim, multilingual)  
Vector index: HNSW (m=16, ef_construction=64)

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
# Install PyTorch with CUDA first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install all dependencies
pip install -r requirements.txt
pip install FlagEmbedding==1.3.3
```

### Step 4 — Configure environment
```powershell
copy .env.example .env
# Edit .env and set: JWT_SECRET_KEY, LLM_BASE_URL, OPENAI_API_KEY
```

### Step 5 — Start the API server
```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Step 6 — Open the app
- Chat UI: http://localhost:8080/
- API Docs: http://localhost:8080/docs
- Health: http://localhost:8080/api/v1/health

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | /api/v1/health | Public | System health check |
| POST | /api/v1/auth/register | Public | Create account |
| POST | /api/v1/auth/login | Public | Login (returns JWT) |
| GET | /api/v1/auth/me | Required | Current user profile |
| POST | /api/v1/query | Required | RAG Q&A (main endpoint) |
| POST | /api/v1/analyze | Required | Glass composition analysis |
| POST | /api/v1/design | Required | Glass composition design |
| POST | /api/v1/troubleshoot | Required | Defect diagnosis |
| GET | /api/v1/conversations | Required | List conversations |
| GET | /api/v1/conversations/{id} | Required | Get conversation messages |
| POST | /api/v1/conversations | Required | Create conversation |
| PATCH | /api/v1/conversations/{id} | Required | Rename conversation |
| DELETE | /api/v1/conversations/{id} | Required | Delete conversation |
| POST | /api/v1/feedback | Required | Submit answer feedback |
| POST | /api/v1/feedback/source | Required | Source relevance feedback |
| POST | /api/v1/ingest | Admin | Document ingestion |

Full Swagger docs: http://localhost:8080/docs

---

## Query Example
```powershell
# Login
$login = Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/auth/login" `
  -ContentType "application/x-www-form-urlencoded" `
  -Body "username=your@email.com&password=yourpassword"
$token = $login.access_token
$headers = @{Authorization = "Bearer $token"}

# Query
Invoke-RestMethod -Method POST -Uri "http://localhost:8080/api/v1/query" `
  -ContentType "application/json" -Headers $headers `
  -Body '{"question": "What is the glass transition temperature of borosilicate glass?", "top_k": 5}'
```

---

## LLM Configuration

Tried in order:
1. **Local vLLM** (fastest, private): Set `LLM_BASE_URL` in `.env`
2. **OpenAI GPT-4o-mini** (cloud fallback): Set `OPENAI_API_KEY` in `.env`
3. **Retrieval-only**: Returns raw retrieved context if no LLM available

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
│   ├── core/
│   │   ├── config.py          ← Centralised settings (pydantic-settings)
│   │   ├── database.py        ← ThreadedConnectionPool (psycopg2)
│   │   └── security.py        ← Auth re-exports
│   ├── auth.py                ← JWT auth (register/login/me)
│   ├── database.py            ← DB shim for backwards compatibility
│   ├── main.py                ← FastAPI app entry point
│   ├── models/
│   │   └── schemas.py         ← Pydantic request/response models
│   └── routers/
│       ├── auth.py            ← /auth endpoints
│       ├── query.py           ← POST /query — full RAG pipeline
│       ├── analyze.py         ← POST /analyze
│       ├── design.py          ← POST /design
│       ├── troubleshoot.py    ← POST /troubleshoot
│       ├── feedback.py        ← POST /feedback
│       ├── conversations.py   ← Conversation CRUD + user memory
│       ├── health.py          ← GET /health
│       └── ingest.py          ← POST /ingest (admin only)
├── ingestion/
│   ├── embedder.py            ← bge-m3 embeddings (ST + instruction prefix)
│   ├── extractor.py           ← PDF, CSV, JSON, DOCX, TXT, NPZ extraction
│   ├── chunker.py             ← Text chunking
│   └── ingest.py              ← Main ingestion pipeline
├── retrieval/
│   ├── retriever.py           ← Hybrid search: dense + sparse + RRF
│   ├── reranker.py            ← bge-reranker-v2-m3 cross-encoder
│   └── llm.py                 ← LLM integration + fallback governance
├── docker/
│   ├── docker-compose.yml     ← PostgreSQL + Redis + pgAdmin
│   └── init.sql               ← Database schema (HNSW index)
├── .env                       ← Configuration (never commit)
├── .env.example               ← Environment template
├── requirements.txt
└── README.md
```

---

## Branches

| Branch | Owner | Description |
|--------|-------|-------------|
| `main` | MTAAI | Default branch |
| `Arjun` | Arjun | Application + RAG pipeline (this branch) |
| `glass-expert-ai` | Engineer Z | LLM fine-tuning + evaluation |