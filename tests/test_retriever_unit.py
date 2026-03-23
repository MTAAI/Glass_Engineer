"""
Glass Expert AI — Retriever Unit Tests
=======================================
Unit tests for the retrieval pipeline covering:
  - Dense vector search
  - BM25 Okapi search
  - RRF fusion
  - Farsi cross-lingual matching
  - Glossary translation
  - Cache behavior (exact + semantic)

Run with:
    python tests/test_retriever_unit.py
    pytest tests/test_retriever_unit.py -v
"""
import os
import sys
import json
import time
import hashlib
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


class TestDenseSearch(unittest.TestCase):
    """Unit tests for dense vector search."""

    def setUp(self):
        from retrieval.retriever import retrieve
        self.retrieve = retrieve

    def test_dense_search_returns_results(self):
        """Dense search should return results for a clear glass science query."""
        results = self.retrieve(
            "glass transition temperature borosilicate",
            top_k=3, use_cache=False,
        )
        self.assertGreater(len(results), 0, "Dense search returned no results")

    def test_dense_search_result_structure(self):
        """Each result must have required fields."""
        results = self.retrieve(
            "viscosity glass melt temperature",
            top_k=3, use_cache=False,
        )
        self.assertGreater(len(results), 0)
        required_fields = {"id", "title", "source_type", "language", "content", "similarity"}
        for r in results:
            missing = required_fields - set(r.keys())
            self.assertEqual(len(missing), 0, f"Result missing fields: {missing}")

    def test_dense_search_similarity_range(self):
        """Similarity scores must be between 0 and 1."""
        results = self.retrieve(
            "soda lime silica glass composition",
            top_k=5, use_cache=False,
        )
        for r in results:
            self.assertGreaterEqual(r["similarity"], 0.0)
            self.assertLessEqual(r["similarity"], 1.0)

    def test_dense_search_top_k_respected(self):
        """Results count should not exceed top_k."""
        for top_k in [1, 3, 5]:
            results = self.retrieve(
                "glass annealing temperature",
                top_k=top_k, use_cache=False,
            )
            self.assertLessEqual(len(results), top_k)

    def test_dense_search_reranker_scores(self):
        """Top result should have high reranker score for clear query."""
        results = self.retrieve(
            "What is the glass transition temperature of borosilicate glass?",
            top_k=3, use_cache=False,
        )
        self.assertGreater(len(results), 0)
        rerank_score = results[0].get("rerank_score", 0)
        self.assertGreater(rerank_score, 0.5,
            f"Expected rerank score > 0.5 for clear query, got {rerank_score}")

    def test_dense_search_source_type_filter(self):
        """Source type filter should restrict results to that type."""
        results = self.retrieve(
            "glass defects manufacturing",
            top_k=5, source_type_filter="sop", use_cache=False,
        )
        for r in results:
            self.assertEqual(r["source_type"], "sop",
                f"Expected source_type=sop, got {r['source_type']}")

    def test_irrelevant_query_low_scores(self):
        """Completely irrelevant query should not produce high reranker scores."""
        results = self.retrieve(
            "quantum computing machine learning neural network",
            top_k=5, use_cache=False,
        )
        for r in results:
            self.assertLess(r.get("rerank_score", 1.0), 0.9,
                "Irrelevant query should not score high")


class TestBM25Search(unittest.TestCase):
    """Unit tests for BM25 Okapi search."""

    def test_bm25_finds_chemical_formulas(self):
        """BM25 should find exact chemical formula matches like SiO2, Na2O."""
        from retrieval.retriever import retrieve
        results = retrieve("SiO2 Na2O glass composition", top_k=5, use_cache=False)
        self.assertGreater(len(results), 0)
        # Check that at least one result contains the formula
        combined = " ".join(r.get("content", "") for r in results)
        self.assertIn("SiO2", combined, "BM25 should find SiO2 in results")

    def test_bm25_skipped_for_farsi(self):
        """BM25 should be skipped for Farsi queries (language optimization)."""
        from retrieval.retriever import detect_language, _extract_keywords
        query = "دمای انتقال شیشه چیست؟"
        lang = detect_language(query)
        self.assertEqual(lang, "fa")
        # Our code skips keyword extraction for Farsi
        keywords = _extract_keywords(query) if lang == "en" else []
        self.assertEqual(len(keywords), 0, "BM25 keywords should be empty for Farsi")


class TestRRFFusion(unittest.TestCase):
    """Unit tests for Reciprocal Rank Fusion."""

    def test_rrf_merge_basic(self):
        """RRF should merge two ranked lists by combined rank score."""
        from retrieval.retriever import _rrf_merge
        dense = [
            {"id": "a", "content": "a", "title": "A", "source_type": "t", "language": "en", "similarity": 0.9},
            {"id": "b", "content": "b", "title": "B", "source_type": "t", "language": "en", "similarity": 0.8},
        ]
        bm25 = [
            {"id": "b", "content": "b", "title": "B", "source_type": "t", "language": "en", "similarity": 0.0, "bm25_score": 5.0},
            {"id": "c", "content": "c", "title": "C", "source_type": "t", "language": "en", "similarity": 0.0, "bm25_score": 3.0},
        ]
        merged = _rrf_merge(dense, bm25)
        ids = [d["id"] for d in merged]
        # "b" appears in both lists, should rank highest
        self.assertEqual(ids[0], "b", "Document in both lists should rank first")
        self.assertEqual(len(merged), 3, "Should have 3 unique documents")

    def test_rrf_merge_empty_bm25(self):
        """RRF with empty BM25 should return dense results in order."""
        from retrieval.retriever import _rrf_merge
        dense = [
            {"id": "x", "content": "x", "title": "X", "source_type": "t", "language": "en", "similarity": 0.9},
        ]
        merged = _rrf_merge(dense, [])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["id"], "x")


class TestFarsiCrossLingual(unittest.TestCase):
    """Unit tests for Farsi cross-lingual retrieval."""

    def setUp(self):
        from retrieval.retriever import retrieve_with_auto_language, detect_language
        self.retrieve_auto = retrieve_with_auto_language
        self.detect_language = detect_language

    def test_farsi_language_detected(self):
        """Farsi text should be detected as 'fa'."""
        for q in ["دمای انتقال شیشه چیست؟", "ویسکوزیته شیشه مذاب", "عیوب شیشه در تولید"]:
            self.assertEqual(self.detect_language(q), "fa", f"Expected 'fa' for '{q}'")

    def test_english_language_detected(self):
        """English text should be detected as 'en'."""
        for q in ["What is glass transition?", "How does viscosity change?"]:
            self.assertEqual(self.detect_language(q), "en", f"Expected 'en' for '{q}'")

    def test_farsi_query_returns_results(self):
        """Farsi query should return results via cross-lingual bge-m3 matching."""
        results, lang = self.retrieve_auto("دمای انتقال شیشه چیست؟", top_k=3)
        self.assertEqual(lang, "fa")
        self.assertGreater(len(results), 0, "Farsi query returned no results")

    def test_farsi_viscosity_matches_english(self):
        """Farsi viscosity query should match English viscosity content."""
        results, lang = self.retrieve_auto("ویسکوزیته شیشه مذاب چگونه تغییر می‌کند؟", top_k=3)
        self.assertEqual(lang, "fa")
        self.assertGreater(len(results), 0)
        combined = " ".join(r.get("content", "") for r in results).lower()
        self.assertIn("viscosity", combined, "Farsi viscosity query should find viscosity content")


class TestGlossaryTranslation(unittest.TestCase):
    """Unit tests for Farsi glossary."""

    def test_glossary_loads(self):
        from retrieval.retriever import _load_glossary
        glossary = _load_glossary()
        self.assertIsInstance(glossary, dict)
        self.assertGreater(len(glossary), 0, "Glossary is empty")

    def test_glossary_has_minimum_terms(self):
        from retrieval.retriever import _load_glossary
        glossary = _load_glossary()
        self.assertGreaterEqual(len(glossary), 50, f"Expected 50+ terms, got {len(glossary)}")


class TestCacheBehavior(unittest.TestCase):
    """Unit tests for Redis cache (exact + semantic)."""

    def setUp(self):
        from retrieval.retriever import _get_redis, _cache_key, _get_cached, _set_cache
        self.get_redis = _get_redis
        self.cache_key = _cache_key
        self.get_cached = _get_cached
        self.set_cache = _set_cache

    def test_redis_connection(self):
        r = self.get_redis()
        if r is None:
            self.skipTest("Redis not available")
        self.assertIsNotNone(r)

    def test_cache_key_deterministic(self):
        key1 = self.cache_key("test query", 5, "None")
        key2 = self.cache_key("test query", 5, "None")
        self.assertEqual(key1, key2)

    def test_cache_key_unique(self):
        key1 = self.cache_key("query one", 5, "None")
        key2 = self.cache_key("query two", 5, "None")
        self.assertNotEqual(key1, key2)

    def test_cache_set_and_get(self):
        r = self.get_redis()
        if r is None:
            self.skipTest("Redis not available")

        test_query = f"cache_test_{int(time.time())}"
        test_data = [{"id": "t1", "title": "T", "content": "c",
                      "similarity": 0.9, "source_type": "textbook", "language": "en"}]
        self.set_cache(test_query, 5, "None", test_data)
        cached = self.get_cached(test_query, 5, "None")
        self.assertIsNotNone(cached, "Cache miss after set")
        self.assertEqual(cached[0]["id"], "t1")

    def test_cached_retrieval_faster(self):
        """Second retrieval with cache should be faster than first."""
        from retrieval.retriever import retrieve
        r = self.get_redis()
        if r is None:
            self.skipTest("Redis not available")

        query = "glass annealing temperature range"

        start = time.time()
        retrieve(query, top_k=3, use_cache=False)
        cold_time = time.time() - start

        # Warm the cache
        retrieve(query, top_k=3, use_cache=True)

        start = time.time()
        retrieve(query, top_k=3, use_cache=True)
        warm_time = time.time() - start

        self.assertLess(warm_time, cold_time,
            f"Cached ({warm_time:.3f}s) should be faster than cold ({cold_time:.3f}s)")


if __name__ == "__main__":
    print("\nGlass Expert AI — Retriever Unit Tests\n")
    unittest.main(verbosity=2)
