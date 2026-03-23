# Glass Expert AI

Bilingual (English + Farsi) RAG-powered glass science assistant for engineers. Built with FastAPI, pgvector, React, and a fine-tuned LLM with OpenAI fallback.

---

## Architecture

```
React UI (:8080) → FastAPI API → JWT Auth
                                → RAG Retrieval (pgvector + bge-large-en-v1.5)
                                → Reranker (cross-encoder)
                                → Redis Cache (:6379)
                                → Fine-tuned LLM (:8000) / OpenAI fallback
                                → PostgreSQL (:5432)
```

## Features

- **RAG Q&A** — retrieval-augmented generation over 54K+ glass science chunks
- **Bilingual** — English and Farsi support with auto-language detection
- **JWT Authentication** — user registration, login, role-based access (admin/engineer)
- **Chat Memory** — conversation history with session management
- **Citations** — source attribution with per-source relevance feedback
- **Specialized Endpoints** — composition analysis, glass design, defect troubleshooting
- **Feedback System** — thumbs up/down with corrected text and source-level ratings
- **Admin Ingestion** — PDF/CSV/DOCX/TXT/JSON/NPZ document pipeline (admin-only)

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
```

### 2. Configure environment

```bash
cp .env.example .env
# Set: DATABASE_URL, REDIS_URL, JWT_SECRET_KEY, OPENAI_API_KEY
```

### 3. Start PostgreSQL and Redis

```bash
cd docker && docker compose up -d && cd ..
```

### 4. Start the LLM model server (optional)

```bash
python model_service/serve.py
# Runs on port 8000 — falls back to OpenAI if unavailable
```

### 5. Start the API server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload
```

### 6. Open the app

Navigate to **http://localhost:8080/** — register an account and start querying.

---

## API Endpoints

All endpoints (except health) require JWT authentication via `Authorization: Bearer <token>`.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/api/v1/health` | Public | System health check |
| POST | `/api/v1/auth/register` | Public | Create account |
| POST | `/api/v1/auth/login` | Public | Login (returns JWT) |
| GET | `/api/v1/auth/me` | Required | Current user profile |
| POST | `/api/v1/query` | Required | RAG Q&A (main endpoint) |
| POST | `/api/v1/analyze` | Required | Composition analysis |
| POST | `/api/v1/design` | Required | Glass composition design |
| POST | `/api/v1/troubleshoot` | Required | Defect diagnosis |
| GET | `/api/v1/conversations` | Required | List conversations |
| GET | `/api/v1/conversations/{id}` | Required | Get conversation messages |
| POST | `/api/v1/conversations` | Required | Create conversation |
| PATCH | `/api/v1/conversations/{id}` | Required | Rename conversation |
| DELETE | `/api/v1/conversations/{id}` | Required | Delete conversation |
| POST | `/api/v1/feedback` | Required | Submit feedback |
| POST | `/api/v1/feedback/source` | Required | Source relevance feedback |
| POST | `/api/v1/ingest` | Admin | Document ingestion |

Swagger docs: **http://localhost:8080/docs**

---

## Project Structure

```
glass-expert-ai/
├── api/
│   ├── main.py               ← FastAPI app + static UI serving
│   ├── auth.py                ← JWT auth (register/login/me)
│   ├── database.py            ← Connection pool management
│   └── routers/
│       ├── query.py           ← RAG Q&A + chat history
│       ├── analyze.py         ← Composition analysis
│       ├── design.py          ← Glass design
│       ├── troubleshoot.py    ← Defect troubleshooting
│       ├── conversations.py   ← Chat session management
│       ├── feedback.py        ← User feedback + source ratings
│       ├── health.py          ← Health check
│       └── ingest.py          ← Document ingestion (admin)
├── frontend/                  ← React + TypeScript + TailwindCSS
│   └── src/
│       ├── App.tsx            ← Main app (auth, chat, sidebar)
│       ├── api/client.ts      ← API client with JWT interceptor
│       └── types/index.ts     ← TypeScript interfaces
├── model_service/
│   └── serve.py               ← Fine-tuned LLM server (port 8000)
├── ingestion/                 ← Document → pgvector pipeline
├── retrieval/
│   ├── retriever.py           ← Dense search + reranker
│   └── llm.py                 ← LLM integration + fallback
├── scripts/
│   ├── training/              ← QLoRA fine-tuning pipeline
│   ├── evaluation/            ← RAG + model evaluation
│   └── cleanup_garbage_chunks.sql
├── data/
│   └── evaluation/            ← Golden eval set + results
├── docker/
│   ├── docker-compose.yml     ← PostgreSQL + Redis + pgAdmin
│   └── init.sql               ← Database schema
└── requirements.txt
```

---

## Services

| Service | Port | Credentials |
|---------|------|-------------|
| API Server | 8080 | JWT auth |
| LLM Server | 8000 | — |
| PostgreSQL | 5432 | glassai / glassai_secret |
| Redis | 6379 | — |
| pgAdmin | 5050 | admin@glassai.com / admin |

---

## Evaluation Results

| Metric | English | Farsi |
|--------|---------|-------|
| GOOD answers | 78% | 0%* |
| RAG retrieval (top-5 hit rate) | ~85% | ~15%* |

*Farsi limited by corpus size (42 Farsi chunks). BGE-M3 multilingual re-embedding in progress to enable cross-lingual retrieval.

---

## Training

QLoRA fine-tuning pipeline for Qwen2.5-14B-Instruct:
- 141K training examples in chat format (system/user/assistant)
- r=64, alpha=128, 4-bit NF4 quantization
- Bilingual (English + Farsi) glass science domain

---

## Tech Stack

- **Backend**: FastAPI, PostgreSQL + pgvector, Redis
- **Frontend**: React, TypeScript, TailwindCSS
- **Embeddings**: bge-large-en-v1.5 (upgrading to bge-m3)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM**: Qwen2.5-14B-Instruct (fine-tuned) + OpenAI gpt-4o-mini fallback
- **Auth**: JWT (python-jose + passlib/bcrypt)
