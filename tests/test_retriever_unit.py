"""
Glass Expert AI — Retriever Unit Tests
=======================================
Unit tests for the retrieval pipeline covering:
  - Dense vector search
  - Farsi cross-lingual matching
  - Glossary translation
  - Cache behavior

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
            top_k=3,
            use_cache=False,
        )
        self.assertGreater(len(results), 0, "Dense search returned no results")

    def test_dense_search_result_structure(self):
        """Each result must have required fields."""
        results = self.retrieve(
            "viscosity glass melt temperature",
            top_k=3,
            use_cache=False,
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
            top_k=5,
            use_cache=False,
        )
        for r in results:
            self.assertGreaterEqual(r["similarity"], 0.0)
            self.assertLessEqual(r["similarity"], 1.0)

    def test_dense_search_top_k_respected(self):
        """Results count should not exceed top_k."""
        for top_k in [1, 3, 5]:
            results = self.retrieve(
                "glass annealing temperature",
                top_k=top_k,
                use_cache=False,
            )
            self.assertLessEqual(len(results), top_k)

    def test_dense_search_reranker_scores(self):
        """Top result should have high reranker score for clear query."""
        results = self.retrieve(
            "What is the glass transition temperature of borosilicate glass?",
            top_k=3,
            use_cache=False,
        )
        self.assertGreater(len(results), 0)
        top = results[0]
        rerank_score = top.get("rerank_score", 0)
        self.assertGreater(rerank_score, 0.5,
            f"Expected rerank score > 0.5 for clear query, got {rerank_score}")

    def test_dense_search_source_type_filter(self):
        """Source type filter should restrict results to that type."""
        results = self.retrieve(
            "glass defects manufacturing",
            top_k=5,
            source_type_filter="sop",
            use_cache=False,
        )
        for r in results:
            self.assertEqual(r["source_type"], "sop",
                f"Expected source_type=sop, got {r['source_type']}")

    def test_dense_search_irrelevant_query(self):
        """Completely irrelevant query should return few or no results above threshold."""
        results = self.retrieve(
            "quantum computing machine learning neural network",
            top_k=5,
            use_cache=False,
        )
        # Either no results or low similarity scores
        for r in results:
            self.assertLess(r.get("rerank_score", 1.0), 0.9,
                "Irrelevant query should not score high on reranker")


class TestFarsiCrossLingual(unittest.TestCase):
    """Unit tests for Farsi cross-lingual retrieval."""

    def setUp(self):
        from retrieval.retriever import retrieve_with_auto_language, detect_language
        self.retrieve_auto = retrieve_with_auto_language
        self.detect_language = detect_language

    def test_farsi_language_detected(self):
        """Farsi text should be detected as 'fa'."""
        farsi_queries = [
            "دمای انتقال شیشه چیست؟",
            "ویسکوزیته شیشه مذاب",
            "عیوب شیشه در تولید",
            "خواص نوری شیشه",
        ]
        for q in farsi_queries:
            lang = self.detect_language(q)
            self.assertEqual(lang, "fa", f"Expected 'fa' for '{q}', got '{lang}'")

    def test_english_language_detected(self):
        """English text should be detected as 'en'."""
        english_queries = [
            "What is the glass transition temperature?",
            "How does viscosity change with temperature?",
            "Glass defects in manufacturing",
        ]
        for q in english_queries:
            lang = self.detect_language(q)
            self.assertEqual(lang, "en", f"Expected 'en' for '{q}', got '{lang}'")

    def test_farsi_query_returns_results(self):
        """Farsi query should return results via cross-lingual bge-m3 matching."""
        results, lang = self.retrieve_auto(
            "دمای انتقال شیشه چیست؟",
            top_k=3,
        )
        self.assertEqual(lang, "fa")
        self.assertGreater(len(results), 0,
            "Farsi query returned no results — bge-m3 cross-lingual matching failed")

    def test_farsi_viscosity_query(self):
        """Farsi viscosity query should match English viscosity content."""
        results, lang = self.retrieve_auto(
            "ویسکوزیته شیشه مذاب چگونه تغییر می‌کند؟",
            top_k=3,
        )
        self.assertEqual(lang, "fa")
        self.assertGreater(len(results), 0)
        # Check that retrieved content is relevant to viscosity
        combined_content = " ".join(r.get("content", "") for r in results).lower()
        self.assertIn("viscosity", combined_content,
            "Farsi viscosity query did not retrieve viscosity content")

    def test_farsi_devitrification_query(self):
        """Farsi devitrification query should match English devitrification content."""
        results, lang = self.retrieve_auto(
            "دیویتریفیکاسیون در شیشه چیست؟",
            top_k=3,
        )
        self.assertEqual(lang, "fa")
        self.assertGreater(len(results), 0)

    def test_farsi_reranker_scores_reasonable(self):
        """Farsi queries should produce reasonable reranker scores via cross-lingual matching."""
        results, _ = self.retrieve_auto(
            "دمای انتقال شیشه چیست؟",
            top_k=3,
        )
        self.assertGreater(len(results), 0)
        top_score = results[0].get("rerank_score", 0)
        self.assertGreater(top_score, 0.3,
            f"Farsi cross-lingual reranker score too low: {top_score}")


class TestGlossaryTranslation(unittest.TestCase):
    """Unit tests for Farsi glossary loading and term matching."""

    def setUp(self):
        from retrieval.retriever import _load_glossary
        self.load_glossary = _load_glossary

    def test_glossary_loads(self):
        """Glossary should load successfully with terms."""
        glossary = self.load_glossary()
        self.assertIsInstance(glossary, dict)
        self.assertGreater(len(glossary), 0,
            "Glossary is empty — check retrieval/glossary_fa_en.json")

    def test_glossary_has_minimum_terms(self):
        """Glossary should have at least 50 terms."""
        glossary = self.load_glossary()
        self.assertGreaterEqual(len(glossary), 50,
            f"Expected 50+ glossary terms, got {len(glossary)}")

    def test_glossary_key_terms_present(self):
        """Key glass science Farsi terms should be in glossary."""
        glossary = self.load_glossary()
        expected_terms = [
            "دمای انتقال شیشه‌ای",  # glass transition temperature
            "ویسکوزیته",              # viscosity
        ]
        for term in expected_terms:
            self.assertIn(term, glossary,
                f"Expected term '{term}' not found in glossary")

    def test_glossary_values_are_english(self):
        """Glossary values should be English strings."""
        glossary = self.load_glossary()
        for fa_term, en_term in list(glossary.items())[:10]:
            self.assertIsInstance(en_term, str)
            self.assertGreater(len(en_term), 0,
                f"Empty English translation for '{fa_term}'")

    def test_glossary_enhances_farsi_query(self):
        """Glossary term should be appended to Farsi query for enhancement."""
        from retrieval.retriever import retrieve_with_auto_language, _load_glossary
        glossary = _load_glossary()

        # Find a Farsi query that contains a glossary term
        test_query = "ویسکوزیته شیشه مذاب چگونه تغییر می‌کند؟"
        has_match = any(fa_term in test_query for fa_term in glossary)
        self.assertTrue(has_match,
            "Test query should contain at least one glossary term")

        # Run retrieval and verify it works
        results, lang = retrieve_with_auto_language(test_query, top_k=3)
        self.assertGreater(len(results), 0,
            "Glossary-enhanced query returned no results")


class TestCacheBehavior(unittest.TestCase):
    """Unit tests for Redis cache behavior."""

    def setUp(self):
        from retrieval.retriever import _get_redis, _cache_key, _get_cached, _set_cache
        self.get_redis = _get_redis
        self.cache_key = _cache_key
        self.get_cached = _get_cached
        self.set_cache = _set_cache

    def test_redis_connection(self):
        """Redis should be accessible."""
        r = self.get_redis()
        if r is None:
            self.skipTest("Redis not available — skipping cache tests")
        self.assertIsNotNone(r)

    def test_cache_key_deterministic(self):
        """Same inputs should always produce same cache key."""
        key1 = self.cache_key("test query", 5, "None")
        key2 = self.cache_key("test query", 5, "None")
        self.assertEqual(key1, key2)

    def test_cache_key_unique_for_different_queries(self):
        """Different queries should produce different cache keys."""
        key1 = self.cache_key("query one", 5, "None")
        key2 = self.cache_key("query two", 5, "None")
        self.assertNotEqual(key1, key2)

    def test_cache_set_and_get(self):
        """Data stored in cache should be retrievable."""
        r = self.get_redis()
        if r is None:
            self.skipTest("Redis not available")

        test_query = f"cache_test_{int(time.time())}"
        test_data = [{"id": "test1", "title": "Test", "content": "test content",
                      "similarity": 0.9, "source_type": "textbook", "language": "en"}]

        self.set_cache(test_query, 5, "None", test_data)
        cached = self.get_cached(test_query, 5, "None")

        self.assertIsNotNone(cached, "Cache miss after set")
        self.assertEqual(len(cached), 1)
        self.assertEqual(cached[0]["id"], "test1")

    def test_cache_miss_returns_none(self):
        """Cache miss should return None."""
        unique_query = f"definitely_not_cached_{time.time()}_xyz"
        cached = self.get_cached(unique_query, 5, "None")
        self.assertIsNone(cached, "Expected cache miss but got a result")

    def test_cached_retrieval_faster(self):
        """Second retrieval with cache should be faster than first."""
        from retrieval.retriever import retrieve

        r = self.get_redis()
        if r is None:
            self.skipTest("Redis not available")

        query = "glass annealing temperature range"

        # First run — no cache
        start = time.time()
        retrieve(query, top_k=3, use_cache=False)
        cold_time = time.time() - start

        # Warm the cache
        retrieve(query, top_k=3, use_cache=True)

        # Second run — should hit cache
        start = time.time()
        results = retrieve(query, top_k=3, use_cache=True)
        warm_time = time.time() - start

        self.assertIsNotNone(results)
        self.assertLess(warm_time, cold_time,
            f"Cached retrieval ({warm_time:.3f}s) should be faster than cold ({cold_time:.3f}s)")


if __name__ == "__main__":
    print("\n" + "🧪 " * 20)
    print("Glass Expert AI — Retriever Unit Tests")
    print("🧪 " * 20 + "\n")
    unittest.main(verbosity=2)