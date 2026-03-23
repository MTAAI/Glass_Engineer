# Glass Expert AI — Llama 8B Deployment

Production-ready RAG system for glass science Q&A, powered by a fine-tuned Llama 3.1 8B model serving 300+ engineers in a glass manufacturing facility.

## System Architecture

```
                                    ┌─────────────────────────────┐
                                    │         Nginx :80           │
                                    │    Reverse Proxy + SSL      │
                                    │   Rate limit, gzip, CORS    │
                                    └─────────────┬───────────────┘
                                                  │
                                    ┌─────────────▼───────────────┐
                                    │     FastAPI Server :8080     │
                                    │   2x uvicorn workers         │
                                    │   JWT auth, rate limiting    │
                                    └──┬──────────┬───────────┬───┘
                                       │          │           │
                          ┌────────────▼──┐  ┌────▼────┐  ┌───▼────────────┐
                          │  PostgreSQL   │  │  Redis  │  │  vLLM :8001    │
                          │  + pgvector   │  │  Cache  │  │  Llama 8B      │
                          │  247K chunks  │  │  2GB    │  │  (merged, GPU) │
                          └───────────────┘  └─────────┘  └────────────────┘
```

## Query Pipeline (End-to-End)

When a user asks a question, this is what happens:

```
User Question
    │
    ▼
1. LANGUAGE DETECTION (langdetect)
    │  → "en" or "fa" (Farsi)
    │  → Farsi queries: glossary-enhanced with English glass terms
    │
    ▼
2. QUERY EMBEDDING (BAAI/bge-m3, 1024-dim)
    │  → Multilingual dense vector via FlagEmbedding
    │  → GPU semaphore limits 4 concurrent embeddings (prevents OOM)
    │
    ▼
3. CACHE CHECK
    │  → Exact cache: MD5(query + top_k + language) → Redis
    │  → Semantic cache: cosine similarity on first 128 dims (threshold 0.92)
    │  → Cache hit? Return immediately. Miss? Continue to retrieval.
    │
    ▼
4. TWO-PASS DENSE RETRIEVAL (pgvector cosine search)
    │  Pass 1: Only textbook + qa_pair sources (priority content)
    │  Pass 2: All source types (papers, manuals, SOPs)
    │  → This ensures textbook definitions surface before niche research papers
    │  → Similarity threshold: 0.42 (raised for bge-m3 precision)
    │  → 20 candidates per pass, deduplicated
    │
    ▼
5. CROSS-ENCODER RERANKING (BAAI/bge-reranker-v2-m3)
    │  → Scores every (query, chunk) pair together
    │  → Much more accurate than bi-encoder cosine alone
    │  → Top 5 chunks selected
    │
    ▼
6. SOURCE TYPE BOOSTING (post-rerank score adjustment)
    │  → textbook: +0.15 | qa_pair: +0.10 | standard: +0.08
    │  → manual: +0.02  | paper: -0.03 (penalize niche research)
    │  → Re-sort by boosted score
    │
    ▼
7. CONTEXT FORMATTING (plain text — matches training format)
    │  → Strip ALL RAG metadata: [Source N] headers, Type/Language/Relevance
    │  → Output pure text paragraphs (the Llama 8B model was trained on plain Q&A)
    │  → Truncate to 12,000 chars (~3000 tokens)
    │
    ▼
8. LLM GENERATION (vLLM → Llama 3.1 8B fine-tuned)
    │  → System prompt: "You are Glass Expert AI..."
    │  → User message: "Reference information:\n{plain_context}\n\nQuestion: {question}"
    │  → Generation params: temp=0.1, rep_penalty=1.12, top_p=0.90, top_k=40
    │  → Max 1024 tokens output
    │  → Degenerate detection: catches repetitive/nonsensical output
    │  → Farsi queries: if local model answers in English, retry with stronger Farsi instruction
    │  → Fallback: OpenAI GPT-4o-mini if local model fails
    │
    ▼
9. POST-PROCESSING
    │  → Validate [Source N] citations (remove invalid references)
    │  → Save Q&A to chat_history (PostgreSQL)
    │  → Log analytics (latency, chunks retrieved, fallback used)
    │
    ▼
Response: answer + cited sources + metadata
```

## Chat Continuity System

The system maintains conversation context across multiple turns using a 3-tier compression strategy:

```
Load up to 20 messages from chat_history table
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Tier 1 (Last 6 messages / 3 turns)             │
│  → Full content preserved                       │
│  → Most relevant for follow-up questions        │
├─────────────────────────────────────────────────┤
│  Tier 2 (Middle messages)                       │
│  → User questions: kept in full                 │
│  → Assistant answers: trimmed to 250 chars      │
│  → Maintains topic flow without token bloat     │
├─────────────────────────────────────────────────┤
│  Tier 3 (Oldest messages)                       │
│  → User questions only (topic markers)          │
│  → Assistant answers: dropped entirely          │
│  → Just enough to know what was discussed       │
└─────────────────────────────────────────────────┘

All history messages are cleaned:
  → KNOWLEDGE BASE CONTEXT blocks stripped
  → Reference information blocks stripped
  → Only the actual Q&A content remains
```

This gives ~2000 tokens of history while maintaining conversation flow for follow-up questions like "what about its thermal properties?" (referring to a glass type mentioned 5 turns ago).

## Retrieval Architecture

### Database: PostgreSQL + pgvector

```
Table: documents_bgem3
├── 247,000 chunks from glass science literature
├── Sources: textbooks, research papers, SOPs, Q&A pairs, manuals
├── Languages: English + Farsi
├── Embeddings: BAAI/bge-m3 (1024-dim, multilingual)
└── Index: HNSW (m=16, ef_construction=200) — 2-5x faster than IVFFlat
```

### Why Two-Pass Retrieval?

The knowledge base has ~5x more research papers than textbooks. Without prioritization, a question like "What is the glass transition temperature of borosilicate glass?" would return niche simulation papers instead of the textbook definition. Two-pass ensures fundamentals come first:

```
Pass 1: SELECT ... WHERE source_type IN ('textbook', 'qa_pair')
        ORDER BY embedding <=> query_vec LIMIT 30
        → Gets foundational definitions

Pass 2: SELECT ... ORDER BY embedding <=> query_vec LIMIT 60
        → Fills remaining slots with papers, manuals, etc.
        → Skips IDs already found in Pass 1
```

### Why Plain Text RAG Format?

The Llama 8B model was fine-tuned on plain Q&A pairs:
```
Q: What is the glass transition temperature of borosilicate glass?
A: The glass transition temperature (Tg) of borosilicate glass is typically...
```

It was NOT trained on structured RAG context with `[Source 1]` headers and metadata. Sending RAG-formatted context caused the model to echo metadata instead of answering. The fix: `_strip_rag_formatting()` removes ALL metadata, producing pure text that matches the training format.

### Caching Strategy

```
Layer 1: Exact Cache (Redis, MD5 hash)
  → Identical query + top_k + language → instant response
  → TTL: 24 hours

Layer 2: Semantic Cache (Redis, cosine similarity)
  → "What is Tg of borosilicate?" ≈ "borosilicate glass transition temperature?"
  → Uses first 128 dims of bge-m3 embedding for fast comparison
  → Threshold: 0.92 (near-identical questions)
  → Max 500 cached queries
  → Avoids redundant embedding + retrieval for paraphrased questions
```

## Model Details

### Llama 3.1 8B — Glass Expert v2

| Property | Value |
|----------|-------|
| Base model | meta-llama/Meta-Llama-3.1-8B-Instruct |
| Fine-tuning | QLoRA (r=64, alpha=128) on 141K glass science Q&A pairs |
| Serving | vLLM with merged weights (15GB on GPU) |
| GPU memory | ~16GB (fits on 24GB RTX with room for batching) |
| Throughput | ~20 tokens/sec generation |
| Context window | 4096 tokens (sufficient for 5-6 sources + answer) |

### Embedding Model: BAAI/bge-m3

| Property | Value |
|----------|-------|
| Dimensions | 1024 |
| Languages | 100+ (English + Farsi natively) |
| Size | 1.7GB |
| Features | Dense + sparse lexical weights (for hybrid search) |
| Runtime | CPU in Docker container (GPU reserved for vLLM) |

### Reranker: BAAI/bge-reranker-v2-m3

| Property | Value |
|----------|-------|
| Type | Cross-encoder (reads query + passage together) |
| Languages | Multilingual (same as bge-m3) |
| Runtime | CPU (fast enough for 20-60 candidates) |
| Max length | 512 tokens per pair |

## Docker Deployment

### Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| **postgres** | pgvector/pgvector:pg16 | 5432 | Knowledge base + chat history + analytics |
| **redis** | redis:7-alpine | 6379 | Query cache (exact + semantic) |
| **vllm** | vllm/vllm-openai | 8001 | Llama 8B model serving (GPU) |
| **api** | glass-expert-ai-llama8b | 8080 | FastAPI backend + React frontend |
| **nginx** | nginx:1.25-alpine | 80 | Reverse proxy, rate limiting, gzip |
| **pgadmin** | dpage/pgadmin4 | 5050 | Database admin (dev profile only) |

### Quick Start

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env with your settings

# 2. Start all services
docker compose up -d

# 3. Wait for vLLM to load model (~2-3 minutes)
docker compose logs -f vllm

# 4. Access the application
# Web UI:    http://localhost
# API Docs:  http://localhost/docs
# pgAdmin:   docker compose --profile dev up pgadmin
```

### Resource Requirements

| Component | CPU | RAM | GPU VRAM | Storage |
|-----------|-----|-----|----------|---------|
| PostgreSQL | 2 cores | 2GB | - | 10GB (247K chunks) |
| Redis | 1 core | 2GB | - | 500MB |
| vLLM (Llama 8B) | 4 cores | 8GB | 16GB | 15GB (model) |
| API Server | 2 cores | 4GB | - | 2GB (embedding model) |
| **Total** | **8 cores** | **16GB** | **16GB** | **28GB** |

## API Endpoints

### Core
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Ask a glass science question (main RAG pipeline) |
| POST | `/api/v1/analyze` | Analyze a glass composition |
| POST | `/api/v1/design` | Design a glass composition for target properties |
| POST | `/api/v1/troubleshoot` | Diagnose glass defects |

### Auth & User
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/login` | JWT authentication |
| POST | `/api/v1/register` | Create new user account |
| GET/POST/DELETE | `/api/v1/user/memory` | User preferences/memory CRUD |

### Conversations
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/conversations` | List user's conversations |
| GET | `/api/v1/conversations/{session_id}` | Load a specific conversation |
| DELETE | `/api/v1/conversations/{session_id}` | Delete a conversation |
| POST | `/api/v1/feedback` | Submit feedback on an answer |

### Admin
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/admin/analytics` | Query performance analytics |
| GET | `/api/v1/health` | System health check |
| POST | `/api/v1/ingest` | Ingest new documents |

## File Structure

```
glass-expert-ai-llama8b/
├── api/                          # FastAPI backend
│   ├── main.py                   # App setup, CORS, rate limiting, middleware
│   ├── auth.py                   # JWT authentication
│   ├── database.py               # Connection pool (psycopg2, min=2, max=20)
│   ├── models/schemas.py         # Pydantic request/response models
│   └── routers/
│       ├── query.py              # Main RAG pipeline endpoint
│       ├── analyze.py            # Composition analysis
│       ├── design.py             # Composition design
│       ├── troubleshoot.py       # Defect diagnosis
│       ├── conversations.py      # Chat history management
│       ├── feedback.py           # User feedback
│       ├── health.py             # Health checks
│       ├── ingest.py             # Document ingestion
│       ├── admin.py              # Admin analytics
│       ├── upload.py             # File upload
│       └── export.py             # Conversation export
├── retrieval/                    # RAG retrieval pipeline
│   ├── retriever.py              # Two-pass dense search + caching
│   ├── llm.py                    # LLM generation + fallback + chat continuity
│   ├── reranker.py               # Cross-encoder reranking
│   └── glossary_fa_en.json       # Farsi → English glass terminology
├── ingestion/                    # Document processing
│   ├── ingest.py                 # Extract → chunk → embed → store
│   ├── extractor.py              # PDF, CSV, DOCX, TXT, JSON extractors
│   ├── chunker.py                # Token-based chunking (512 tokens, 64 overlap)
│   └── embedder.py               # bge-m3 embedding (FlagEmbedding + fallback)
├── frontend/                     # React chat UI
│   ├── src/App.tsx               # Main app (chat, sidebar, settings)
│   ├── src/api/client.ts         # API client functions
│   └── src/types/index.ts        # TypeScript interfaces
├── docker/
│   ├── init.sql                  # Database schema (all tables + indexes)
│   └── nginx.conf                # Reverse proxy config
├── scripts/
│   └── evaluation/               # Eval scripts + golden test set
├── data/evaluation/results/      # Evaluation results (3.80/5 avg)
├── Dockerfile                    # Multi-stage build
├── docker-compose.yml            # Full production stack
├── requirements-docker.txt       # Runtime deps only (no training)
└── .env.example                  # Environment variables template
```

## Evaluation Results

**English evaluation (50 questions):** Average score 3.80/5

| Score | Count | Description |
|-------|-------|-------------|
| 5/5 | 18 | Perfect — accurate with specific values |
| 4/5 | 12 | Good — correct but missing some detail |
| 3/5 | 13 | Acceptable — partially correct |
| 2/5 | 7 | Poor — missing key information (data gaps in knowledge base) |

The 7 questions scoring 2/5 are data quality issues (NiS inclusions, chemical durability) where the knowledge base lacks textbook content — not model or retrieval problems.

## Bilingual Support (English + Farsi)

```
English Query → bge-m3 embedding → dense search → BM25 keyword search → RRF fusion → rerank → LLM
Farsi Query   → bge-m3 embedding → dense search only (BM25 skipped) → rerank → LLM (Farsi prompt)
```

- **BM25 is English-only** — Farsi tokenization needs different handling; bge-m3 dense search handles Farsi natively
- **Farsi queries** get glossary-enhanced with English glass terms before embedding
- **Farsi responses** use a dedicated system prompt with explicit Farsi output instructions
- **Language retry** — if local model responds in English for a Farsi query, retries with stronger Farsi instruction
- **OpenAI fallback** — GPT-4o-mini handles Farsi fluently as a safety net

## Security

- JWT authentication with configurable expiration
- Non-root Docker user (`appuser`)
- Rate limiting (30 req/min per IP)
- CORS restricted to configured origins
- `.env` excluded from git (contains API keys)
- Nginx security headers (X-Frame-Options, X-Content-Type-Options)
- Database connection pooling with max 20 connections
