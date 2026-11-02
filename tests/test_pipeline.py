"""
tests/test_pipeline.py
----------------------
Unit tests for the end-to-end RAG pipeline and context engineering.
"""

import unittest
from pathlib import Path
from src.pipeline import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestRAGPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Initialize pipeline in local mode for fast unit tests
        cls.pipeline = RAGPipeline(mode="local")

    def test_pipeline_initialization(self):
        self.assertIsNotNone(self.pipeline.bm25)
        self.assertIsNotNone(self.pipeline.dense)
        self.assertIsNotNone(self.pipeline.reranker)
        self.assertGreater(len(self.pipeline.chunk_lookup), 9000)

    def test_retrieve_returns_top_k(self):
        query = "What parameters does itertools.batched() accept?"
        chunks, timings = self.pipeline.retrieve(query, top_k=5)

        self.assertEqual(len(chunks), 5)
        self.assertIn("stage1_hybrid_ms", timings)
        self.assertIn("stage2_rerank_ms", timings)
        self.assertIn("total_latency_ms", timings)

        # Check chunk fields
        top_chunk = chunks[0]
        self.assertEqual(top_chunk["rank"], 1)
        self.assertIn("chunk_id", top_chunk)
        self.assertIn("final_score", top_chunk)
        self.assertIn("doc_path", top_chunk)
        self.assertIn("text", top_chunk)

        # Ensure scores are sorted descending
        scores = [c["final_score"] for c in chunks]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_u_shaped_context_formatting(self):
        mock_chunks = [
            {"rank": 1, "chunk_id": "c1", "doc_path": "doc1.html", "breadcrumbs": ["A"], "text": "Content 1"},
            {"rank": 2, "chunk_id": "c2", "doc_path": "doc2.html", "breadcrumbs": ["B"], "text": "Content 2"},
            {"rank": 3, "chunk_id": "c3", "doc_path": "doc3.html", "breadcrumbs": ["C"], "text": "Content 3"},
        ]

        formatted = self.pipeline.format_context(mock_chunks, order_strategy="u_shaped")

        self.assertIn("<official_python_312_documentation>", formatted)
        self.assertIn('id="c1"', formatted)
        self.assertIn('id="c2"', formatted)
        self.assertIn('id="c3"', formatted)

        # U-shaped ordering: rank 1 first, rank 3 middle, rank 2 last
        pos_c1 = formatted.find('id="c1"')
        pos_c3 = formatted.find('id="c3"')
        pos_c2 = formatted.find('id="c2"')

        self.assertTrue(pos_c1 < pos_c3 < pos_c2, "U-shaped order should place #1 at top and #2 at bottom")

    def test_query_method_end_to_end(self):
        query = "How to test floating point numbers with math.isclose?"
        res = self.pipeline.query(query, top_k=3)

        self.assertEqual(res["query"], query)
        self.assertEqual(len(res["chunks"]), 3)
        self.assertIn("llm_context_prompt", res)
        self.assertIn("library_math::api_math.isclose::0", [c["chunk_id"] for c in res["chunks"]])

    def test_section_deduplication(self):
        query = "How to test floating point numbers with math.isclose?"
        # Request top 10 with max_chunks_per_section=1
        chunks, _ = self.pipeline.retrieve(query, top_k=6, max_chunks_per_section=1)
        sec_counts = {}
        for c in chunks:
            sec_key = (c["doc_path"], tuple(c["breadcrumbs"]))
            sec_counts[sec_key] = sec_counts.get(sec_key, 0) + 1

        # Every section must appear at most 1 time
        for sec, count in sec_counts.items():
            self.assertLessEqual(count, 1, f"Section {sec} exceeded max count 1 (got {count})")

    def test_adaptive_score_slicing(self):
        query = "How to test floating point numbers with math.isclose?"
        # Query in default adaptive mode (no explicit top_k)
        chunks, _ = self.pipeline.retrieve(query)

        # Must respect bounds [min_chunks=10, max_chunks=16]
        self.assertGreaterEqual(len(chunks), self.pipeline.min_chunks)
        self.assertLessEqual(len(chunks), self.pipeline.max_chunks)

        # For any chunk selected past min_chunks, score must be >= score_floor
        for c in chunks[self.pipeline.min_chunks:]:
            self.assertGreaterEqual(c["final_score"], self.pipeline.score_floor)


if __name__ == "__main__":
    unittest.main()

