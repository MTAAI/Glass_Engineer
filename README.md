# Glass Expert AI — Phase 3: RAG API + React Chat UI

A RAG-powered glass science assistant for engineers. Built with FastAPI, pgvector, and a React chat interface. Powered by Engineer Z's `glass-expert` vLLM server.

---

## What's New in Phase 3

| Component | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Ingestion | PDF only | PDF, CSV, NPZ, TXT, DOCX, JSON | Same |
| Retrieval | Script only | REST API endpoint | Same |
| Answer generation | None | LLM-powered | `glass-expert` vLLM (Engineer Z) |
| User interface | None | None | React chat UI at `/` |
| API docs | None | None | Swagger at `/docs` (dev only) |
| Health monitoring | None | `/api/v1/health` | `/api/v1/health` |
| Specialized endpoints | None | None | `/analyze`, `/design`, `/troubleshoot`, `/feedback` |

---

## Quick Start

### Step 1 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 2 — Configure environment

Copy `.env.example` to `.env` and fill in your values. The LLM is already configured:

```env
LLM_BASE_URL=https://michelle-angel-vote-extended.trycloudflare.com/v1
LLM_MODEL=glass-expert
LLM_API_KEY=token-glass-ai
```

### Step 3 — Start PostgreSQL and Redis

```powershell
# Option A: Docker
cd docker && docker compose up -d && cd ..

# Option B: Native (Linux/Mac)
sudo service postgresql start
sudo service redis-server start
```

### Step 4 — Start the API server

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Step 5 — Open the Chat UI

Navigate to: **http://localhost:8080/**

You will see the Glass Expert AI React chat interface. Ask any glass science question and get answers powered by the `glass-expert` LLM.

---

## API Endpoints

### `GET /api/v1/health`
Check the status of all system components (database, Redis, embedding model).

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

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Your glass science question |
| `top_k` | int | 5 | Number of chunks to retrieve (1–20) |
| `source_type` | string | null | Filter by: `textbook`, `paper`, `sop`, `standard`, `manual` |
| `language` | string | null | Force language: `en` or `fa`. Auto-detected if not set. |

### `POST /api/v1/analyze`
Analyze a glass composition and predict properties.

### `POST /api/v1/design`
Design a glass composition for target properties.

### `POST /api/v1/troubleshoot`
Diagnose glass defects and get root cause analysis.

### `POST /api/v1/ingest`
Ingest a new document into the knowledge base via API.

---

## LLM Configuration

The system uses Engineer Z's `glass-expert` vLLM server:

```env
LLM_BASE_URL=https://michelle-angel-vote-extended.trycloudflare.com/v1
LLM_MODEL=glass-expert
LLM_API_KEY=token-glass-ai
```

---

## Ingesting Your 15 GB Dataset

Once your real data is ready, organize it and run:

```powershell
# Clear sample data first
python -c "import psycopg2, os; conn = psycopg2.connect(os.getenv('DATABASE_URL')); conn.cursor().execute('TRUNCATE TABLE documents, ingestion_log RESTART IDENTITY CASCADE;'); conn.commit()"

# Ingest by folder (runs overnight)
python 03_build_rag.py --folder data/real_data/textbooks/ --type textbook
python 03_build_rag.py --folder data/real_data/papers/    --type paper
python 03_build_rag.py --folder data/real_data/sops/      --type sop
python 03_build_rag.py --folder data/real_data/standards/ --type standard
```

Supported file types: **PDF, CSV, NPZ, TXT, DOCX, JSON**

---

## Project Structure

```
glass-expert-ai/
├── api/
│   ├── main.py              ← FastAPI app + React UI serving
│   ├── routers/
│   │   ├── query.py         ← POST /api/v1/query (RAG pipeline)
│   │   ├── health.py        ← GET  /api/v1/health
│   │   ├── ingest.py        ← POST /api/v1/ingest
│   │   ├── analyze.py       ← POST /api/v1/analyze
│   │   ├── design.py        ← POST /api/v1/design
│   │   ├── troubleshoot.py  ← POST /api/v1/troubleshoot
│   │   └── feedback.py      ← POST /api/v1/feedback
│   └── models/
│       └── schemas.py       ← Pydantic request/response models
├── frontend/
│   ├── src/                 ← React + TypeScript + TailwindCSS
│   └── dist/                ← Built frontend (served by FastAPI)
├── ingestion/
│   ├── extractor.py         ← Multi-format text extraction
│   ├── chunker.py           ← Text chunking
│   ├── embedder.py          ← BAAI/bge-large-en-v1.5 embeddings
│   └── ingest.py            ← Main ingestion pipeline
├── retrieval/
│   ├── retriever.py         ← pgvector similarity search + Redis cache
│   └── llm.py               ← LLM integration (vLLM / OpenAI)
├── data/
│   └── sample_docs/         ← 3 sample documents (26 chunks)
├── docker/
│   ├── docker-compose.yml
│   └── init.sql
├── .env                     ← Configuration (DB, Redis, LLM settings)
├── requirements.txt
├── 03_build_rag.py          ← Bulk ingestion script
└── README.md
```
