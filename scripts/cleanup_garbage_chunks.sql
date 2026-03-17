-- ============================================================
-- Glass Expert AI — Garbage Chunk Cleanup Script
-- Run against the glass_expert_ai database
-- ============================================================
-- ALWAYS run with BEGIN/COMMIT and inspect counts before committing.

BEGIN;

-- ── 1. Delete junk chunks: too short (<20 words) ──────────────────────────────
-- These are headers, TOC entries, page numbers, etc. that got chunked.
DELETE FROM documents
WHERE id IN (
    SELECT id FROM documents
    WHERE array_length(regexp_split_to_array(trim(content), '\s+'), 1) < 20
      AND source_type NOT IN ('qa_pair')  -- QA pairs can be short
);
-- Check: SELECT count(*) FROM documents WHERE array_length(regexp_split_to_array(trim(content), '\s+'), 1) < 20 AND source_type NOT IN ('qa_pair');

-- ── 2. Delete chunks with >50% non-alpha characters ──────────────────────────
-- These are tables of numbers, garbled OCR, binary fragments, etc.
DELETE FROM documents
WHERE id IN (
    SELECT id FROM documents
    WHERE length(regexp_replace(content, '[^a-zA-Z\u0600-\u06FF ]', '', 'g')) < length(content) * 0.5
      AND length(content) > 50  -- only check chunks with some content
      AND source_type NOT IN ('qa_pair')
);

-- ── 3. Deduplicate exact content matches ──────────────────────────────────────
-- Keep the first (oldest) copy of each duplicate chunk per title.
DELETE FROM documents
WHERE id IN (
    SELECT id FROM (
        SELECT id,
               ROW_NUMBER() OVER (
                   PARTITION BY md5(content), title
                   ORDER BY created_at ASC
               ) AS rn
        FROM documents
    ) ranked
    WHERE rn > 1
);

-- ── 4. Delete near-duplicate chunks (same title, >95% content overlap) ────────
-- This catches the 538 double-ingested domain chunks.
DELETE FROM documents a
USING documents b
WHERE a.id > b.id
  AND a.title = b.title
  AND a.source_type = b.source_type
  AND length(a.content) > 50
  AND abs(length(a.content) - length(b.content)) < length(a.content) * 0.05
  AND left(a.content, 200) = left(b.content, 200);

-- ── 5. Delete empty or null content ──────────────────────────────────────────
DELETE FROM documents
WHERE content IS NULL
   OR trim(content) = '';

-- ── Summary ─────────────────────────────────────────────────────────────────
-- Run these BEFORE commit to inspect:
-- SELECT 'remaining' AS status, count(*) FROM documents;
-- SELECT source_type, count(*) FROM documents GROUP BY source_type ORDER BY count(*) DESC;
-- SELECT language, count(*) FROM documents GROUP BY language;

-- Uncomment to apply:
-- COMMIT;

-- Safety: rollback by default until you verify the counts
ROLLBACK;
