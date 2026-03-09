-- ============================================================
-- Glass Expert AI — Database Initialization Script
-- PostgreSQL 16 + pgvector
-- ============================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search

-- ============================================================
-- USERS TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name       VARCHAR(255),
    role            VARCHAR(50) DEFAULT 'engineer',
    plant_location  VARCHAR(100),
    language_pref   VARCHAR(10) DEFAULT 'en',
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- DOCUMENTS TABLE (Knowledge Base Chunks)
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL,
    source_type VARCHAR(50) NOT NULL
                CHECK (source_type IN ('textbook','paper','sop','qa_pair','manual','standard')),
    language    VARCHAR(10) DEFAULT 'en'
                CHECK (language IN ('en','fa')),
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{}',
    embedding   vector(1024),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Vector similarity search index (cosine distance)
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Full-text search index
CREATE INDEX IF NOT EXISTS documents_content_fts_idx
    ON documents
    USING gin(to_tsvector('english', content));

-- Metadata and filter indexes
CREATE INDEX IF NOT EXISTS documents_source_type_idx ON documents (source_type);
CREATE INDEX IF NOT EXISTS documents_language_idx    ON documents (language);
CREATE INDEX IF NOT EXISTS documents_title_idx       ON documents (title);

-- ============================================================
-- CHAT HISTORY TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_history (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id  UUID NOT NULL DEFAULT gen_random_uuid(),
    role        VARCHAR(20) NOT NULL CHECK (role IN ('user','assistant','system')),
    content     TEXT NOT NULL,
    sources     JSONB DEFAULT '[]',
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chat_history_user_id_idx    ON chat_history (user_id);
CREATE INDEX IF NOT EXISTS chat_history_session_id_idx ON chat_history (session_id);
CREATE INDEX IF NOT EXISTS chat_history_created_at_idx ON chat_history (created_at DESC);

-- ============================================================
-- FEEDBACK TABLE
-- ============================================================
CREATE TABLE IF NOT EXISTS feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id         UUID NOT NULL REFERENCES chat_history(id) ON DELETE CASCADE,
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    rating          SMALLINT NOT NULL CHECK (rating IN (-1, 1)),
    corrected_text  TEXT,
    comment         TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS feedback_chat_id_idx ON feedback (chat_id);
CREATE INDEX IF NOT EXISTS feedback_rating_idx  ON feedback (rating);

-- ============================================================
-- SOURCE FEEDBACK TABLE (Which documents were relevant?)
-- ============================================================
CREATE TABLE IF NOT EXISTS source_feedback (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feedback_id UUID NOT NULL REFERENCES feedback(id) ON DELETE CASCADE,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    is_relevant BOOLEAN NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- USER MEMORY TABLE (Long-term conversation memory)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_memory (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    memory_type VARCHAR(50) NOT NULL,   -- 'composition', 'furnace', 'topic', 'preference'
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS user_memory_user_id_idx ON user_memory (user_id);

-- ============================================================
-- INGESTION LOG TABLE (Track what has been ingested)
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_log (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_name       TEXT NOT NULL,
    file_path       TEXT,
    source_type     VARCHAR(50),
    language        VARCHAR(10),
    chunk_count     INTEGER DEFAULT 0,
    status          VARCHAR(20) DEFAULT 'completed'
                    CHECK (status IN ('completed','failed','processing')),
    error_message   TEXT,
    ingested_at     TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- HELPER FUNCTION: Update updated_at timestamp automatically
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_memory_updated_at
    BEFORE UPDATE ON user_memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================
-- VERIFICATION
-- ============================================================
DO $$
BEGIN
    RAISE NOTICE '✅ Glass Expert AI database initialized successfully.';
    RAISE NOTICE '   Tables: documents, users, chat_history, feedback, source_feedback, user_memory, ingestion_log';
    RAISE NOTICE '   Extensions: vector (pgvector), uuid-ossp, pg_trgm';
END $$;
