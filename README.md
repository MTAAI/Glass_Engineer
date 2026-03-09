# Glass Expert AI — RAG + Fine-tuned LLM + OpenAI Fallback

A RAG-powered glass science assistant for engineers. Built with FastAPI, pgvector, and a React chat interface. Powered by a fine-tuned Llama 3.1 8B glass-expert model.

---

## Architecture

```
User → React UI (8080) → FastAPI API → RAG retrieval (pgvector)
                                      → Fine-tuned LLM (8000)
                                      → OpenAI fallback
```

## Quick Start

### Step 1 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 2 — Configure environment

```powershell
cp .env.example .env
# Edit .env with your values
```

### Step 3 — Start PostgreSQL and Redis

```powershell
cd docker && docker compose up -d && cd ..
```

### Step 4 — Start the LLM model server

```powershell
python model_service/serve.py
# Runs on port 8000
```

### Step 5 — Start the API server

```powershell
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### Step 6 — Open the Chat UI

Navigate to: **http://localhost:8080/**

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | System health check |
| POST | `/api/v1/query` | RAG Q&A (main endpoint) |
| POST | `/api/v1/analyze` | Composition analysis |
| POST | `/api/v1/design` | Composition design |
| POST | `/api/v1/troubleshoot` | Defect diagnosis |
| POST | `/api/v1/feedback` | User feedback |
| POST | `/api/v1/ingest` | Document ingestion |

Swagger docs: **http://localhost:8080/docs**

---

## Project Structure

```
glass-expert-ai/
├── model_service/
│   └── serve.py              ← Fine-tuned LLM server (port 8000)
├── api/
│   ├── main.py               ← FastAPI app + React UI (port 8080)
│   └── routers/              ← API endpoints
├── frontend/                 ← React + TypeScript + TailwindCSS
├── ingestion/                ← Document → pgvector pipeline
├── retrieval/                ← RAG search + LLM integration
├── scripts/
│   ├── qa_generation/        ← Q&A dataset generation
│   ├── training/             ← Fine-tuning scripts
│   └── evaluation/           ← Model evaluation
├── data/
│   ├── evaluation/           ← Golden eval set + results
│   └── sample_docs/          ← Sample documents
├── docker/
│   ├── docker-compose.yml    ← PostgreSQL + Redis
│   └── init.sql              ← Database schema
└── requirements.txt
```

---

## Services

| Service | Host | Port | Credentials |
|---------|------|------|-------------|
| LLM Server | localhost | 8000 | — |
| API Server | localhost | 8080 | — |
| PostgreSQL | localhost | 5432 | glassai / glassai_secret |
| Redis | localhost | 6379 | — |
| pgAdmin | localhost | 5050 | admin@glassai.com / admin |
