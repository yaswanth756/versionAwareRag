"""
tests/test_app.py
-----------------
Unit tests for the FastAPI web server and endpoints in src/app.py.
"""

import unittest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from src.app import app, get_generator
import src.app as app_module


class TestApp(unittest.TestCase):

    def setUp(self):
        # Create a mock RAGGenerator
        self.mock_generator = MagicMock()
        self.mock_generator.model_name = "gemini-2.5-flash"
        self.mock_generator.location = "asia-south1"
        self.mock_generator.project_id = "test-project"
        self.mock_generator.pipeline.reranker_backend = "BAAI/bge-reranker-large (Remote)"

        # Inject mock into app_module
        app_module.generator_instance = self.mock_generator

        # Create test client without triggering real lifespan startup
        self.client = TestClient(app, raise_server_exceptions=True)

    def tearDown(self):
        app_module.generator_instance = None

    def test_get_index_html(self):
        """Verify GET / returns 200 and loads the HTML test interface."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Python 3.12 Documentation RAG System", response.text)
        self.assertIn("query-input", response.text)
        self.assertIn("Ask Python 3.12 Docs", response.text)

    def test_health_check(self):
        """Verify GET /health returns status and model info."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["model"], "gemini-2.5-flash")
        self.assertEqual(data["location"], "asia-south1")

    def test_query_endpoint_success(self):
        """Verify POST /api/query processes query and returns structured result."""
        self.mock_generator.generate.return_value = {
            "query": "What is itertools.batched?",
            "answer": "itertools.batched() groups elements into tuples of length n [Doc 1].",
            "is_out_of_domain": False,
            "cited_doc_indices": [1],
            "cited_chunks": [
                {
                    "rank": 1,
                    "doc_path": "library/itertools.html",
                    "section_breadcrumbs": ["Itertools", "batched"],
                    "ce_score": 0.985,
                    "content": "itertools.batched(iterable, n)...",
                }
            ],
            "all_retrieved_chunks": [
                {
                    "rank": 1,
                    "doc_path": "library/itertools.html",
                    "section_breadcrumbs": ["Itertools", "batched"],
                    "ce_score": 0.985,
                    "bm25_score": 14.5,
                    "dense_score": 0.88,
                    "content": "itertools.batched(iterable, n)...",
                }
            ],
            "model": "gemini-2.5-flash",
            "location": "asia-south1",
            "timings": {
                "stage1_hybrid_ms": 32.5,
                "stage2_rerank_ms": 45.1,
                "retrieval_total_ms": 77.6,
                "generation_ms": 410.2,
                "total_e2e_ms": 487.8,
            },
        }

        payload = {"query": "What is itertools.batched?"}
        response = self.client.post("/api/query", json=payload)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data["query"], "What is itertools.batched?")
        self.assertIn("itertools.batched() groups elements", data["answer"])
        self.assertFalse(data["is_out_of_domain"])
        self.assertEqual(data["cited_doc_indices"], [1])
        self.assertEqual(len(data["cited_chunks"]), 1)
        self.assertEqual(len(data["all_retrieved_chunks"]), 1)
        self.assertEqual(data["timings"]["total_e2e_ms"], 487.8)

    def test_query_endpoint_empty_query(self):
        """Verify POST /api/query rejects empty query."""
        response = self.client.post("/api/query", json={"query": "   "})
        self.assertEqual(response.status_code, 400)
        self.assertIn("Query cannot be empty", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()

