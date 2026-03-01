# Glass Expert AI — Phase 2: RAG API

This is the Phase 2 upgrade of the Glass Expert AI system. It adds a **FastAPI backend** that exposes the full RAG pipeline (retrieve + generate) as a REST API, ready to connect to any frontend or chat interface.

---

## What's New in Phase 2

| Component | Phase 1 | Phase 2 |
|---|---|---|
| Ingestion | PDF only | PDF, CSV, NPZ, TXT, DOCX, JSON |
| Retrieval | Script only | REST API endpoint |
| Answer generation | None | LLM-powered (vLLM / OpenAI) |
| API docs | None | Swagger UI at `/docs` |
| Health monitoring | None | `/api/v1/health` endpoint |

---

## Quick Start

### Step 1 — Make sure Docker is running

```powershell
cd docker
docker compose up -d
cd ..
```

### Step 2 — Activate virtual environment

```powershell
.\venv\Scripts\activate
```

### Step 3 — Install new Phase 2 dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Start the API server

```powershell
.\start_api.ps1
```

Or manually:

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Step 5 — Open the interactive API docs

Navigate to: **http://localhost:8080/docs**

You will see the full Swagger UI where you can test every endpoint directly in the browser.

---

## API Endpoints

### `GET /api/v1/health`
Check the status of all system components (database, Redis, embedding model).

**Example response:**
```json
{
  "status": "healthy",
  "database": "healthy",
  "redis": "healthy",
  "embedding_model": "ready (BAAI/bge-m3)",
  "total_documents": 3,
  "total_chunks": 12,
  "version": "2.0.0"
}
```

---

### `POST /api/v1/query`
Ask a glass science question. Returns an LLM-generated answer with cited sources.

**Request body:**
```json
{
  "question": "What is the glass transition temperature of borosilicate glass?",
  "top_k": 5,
  "source_type": "textbook"
}
```

**Response:**
```json
{
  "question": "What is the glass transition temperature of borosilicate glass?",
  "answer": "Borosilicate glass has a glass transition temperature (Tg) of approximately 525°C...",
  "sources": [
    {
      "title": "01_glass_transition_thermal_properties",
      "source_type": "textbook",
      "language": "en",
      "similarity": 0.6788,
      "content_preview": "..."
    }
  ],
  "language_detected": "en",
  "retrieval_time_ms": 62.5,
  "total_chunks_searched": 1,
  "model_used": "meta-llama/Meta-Llama-3-8B-Instruct"
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Your glass science question |
| `top_k` | int | 5 | Number of chunks to retrieve (1–20) |
| `source_type` | string | null | Filter by: `textbook`, `paper`, `sop`, `standard`, `manual` |
| `language` | string | null | Force language: `en` or `fa`. Auto-detected if not set. |

---

### `POST /api/v1/ingest`
Ingest a new document into the knowledge base via API.

**Request body:**
```json
{
  "file_path": "C:/path/to/your/document.pdf",
  "source_type": "textbook"
}
```

---

## LLM Configuration

The system supports three LLM modes, tried in order:

1. **Local vLLM** (fastest, private): Set `LLM_BASE_URL` in `.env` to your vLLM server URL
2. **OpenAI GPT** (cloud fallback): Set `OPENAI_API_KEY` in `.env`
3. **Retrieval-only** (no LLM): If neither is available, returns the raw retrieved context

To set up local vLLM with Llama-3-8B:
```powershell
pip install vllm
python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3-8B-Instruct --port 8000
```

---

## Ingesting Your 15GB Dataset

Once your real data arrives, organize it and run:

```powershell
# Clear sample data first
docker exec glass_ai_postgres psql -U glassai -d glass_expert_ai -c "TRUNCATE TABLE documents, ingestion_log RESTART IDENTITY CASCADE;"

# Ingest by folder (runs overnight)
python ingestion/ingest.py data/real_data/textbooks/ --type textbook
python ingestion/ingest.py data/real_data/papers/ --type paper
python ingestion/ingest.py data/real_data/sops/ --type sop
python ingestion/ingest.py data/real_data/standards/ --type standard
```

Supported file types: **PDF, CSV, NPZ, TXT, DOCX, JSON**

---

## Project Structure

```
glass-expert-ai/
├── api/
│   ├── main.py              ← FastAPI app entry point
│   ├── routers/
│   │   ├── query.py         ← POST /api/v1/query (RAG pipeline)
│   │   ├── health.py        ← GET  /api/v1/health
│   │   └── ingest.py        ← POST /api/v1/ingest
│   └── models/
│       └── schemas.py       ← Pydantic request/response models
├── ingestion/
│   ├── extractor.py         ← Multi-format text extraction
│   ├── chunker.py           ← Text chunking
│   ├── embedder.py          ← BAAI/bge-m3 GPU embeddings
│   └── ingest.py            ← Main ingestion pipeline
├── retrieval/
│   ├── retriever.py         ← pgvector similarity search + Redis cache
│   └── llm.py               ← LLM integration (vLLM / OpenAI)
├── tests/
│   ├── test_retrieval.py    ← Phase 1 retrieval tests
│   └── test_api.py          ← Phase 2 API tests
├── docker/
│   ├── docker-compose.yml
│   └── init.sql
├── .env                     ← Configuration (DB, Redis, LLM settings)
├── requirements.txt
├── start_api.ps1            ← One-click API start script
└── README.md
```
