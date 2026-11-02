"""
tests/test_generator.py
------------------------
Unit tests for the RAGGenerator citation extraction and prompt contract.
"""

import unittest
from unittest.mock import MagicMock, patch
from src.generator import RAGGenerator, SYSTEM_INSTRUCTION


class TestRAGGenerator(unittest.TestCase):

    def setUp(self):
        # Create generator with mock client and mock pipeline
        self.mock_pipeline = MagicMock()
        self.mock_client = MagicMock()

        with patch("google.genai.Client", return_value=self.mock_client):
            self.generator = RAGGenerator(
                project_id="test-project",
                location="asia-south1",
                model_name="gemini-2.5-flash",
                pipeline=self.mock_pipeline,
            )

    def test_citation_extraction(self):
        text = (
            "The `math.isclose()` function tests floating-point proximity [Doc 1]. "
            "It accepts `rel_tol` and `abs_tol` parameters [Doc 1] [Doc 2]. "
            "For complex numbers, see `cmath.isclose()` [Doc 5]."
        )
        citations = self.generator._extract_citations(text)
        self.assertEqual(citations, [1, 2, 5])

    def test_citation_extraction_empty(self):
        text = "This is a response with no citations."
        citations = self.generator._extract_citations(text)
        self.assertEqual(citations, [])

    def test_system_prompt_grounding_rules(self):
        self.assertIn("Strict Grounding", SYSTEM_INSTRUCTION)
        self.assertIn("Refusal / Abstention Policy", SYSTEM_INSTRUCTION)
        self.assertIn("Inline Document Citations", SYSTEM_INSTRUCTION)
        self.assertIn("<official_python_312_documentation>", SYSTEM_INSTRUCTION)

    def test_generate_non_streaming(self):
        # Mock retrieval
        self.mock_pipeline.query.return_value = {
            "query": "Test question?",
            "llm_context_prompt": "<official_python_312_documentation></official_python_312_documentation>",
            "timings": {"total_latency_ms": 100.0, "stage1_hybrid_ms": 60.0, "stage2_rerank_ms": 40.0},
            "chunks": [
                {"rank": 1, "chunk_id": "c1", "doc_path": "doc1.html"},
                {"rank": 2, "chunk_id": "c2", "doc_path": "doc2.html"},
            ],
        }

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.text = "Answer supported by official documentation [Doc 1]."
        self.mock_client.models.generate_content.return_value = mock_response

        res = self.generator.generate("Test question?", stream=False)

        self.assertEqual(res["query"], "Test question?")
        self.assertEqual(res["answer"], "Answer supported by official documentation [Doc 1].")
        self.assertEqual(res["cited_doc_indices"], [1])
        self.assertEqual(len(res["cited_chunks"]), 1)
        self.assertEqual(res["cited_chunks"][0]["chunk_id"], "c1")
        self.assertIn("generation_ms", res["timings"])
        self.assertIn("total_e2e_ms", res["timings"])
        self.assertFalse(res["is_out_of_domain"])

    def test_out_of_domain_short_circuit_remote(self):
        # Top chunk has extremely low CE score (out of domain)
        self.mock_pipeline.query.return_value = {
            "query": "What is the capital of France?",
            "reranker_backend": "Remote GPU (BAAI/bge-reranker-large on Tesla T4)",
            "timings": {"total_latency_ms": 50.0, "stage1_hybrid_ms": 30.0, "stage2_rerank_ms": 20.0},
            "chunks": [
                {"rank": 1, "chunk_id": "c1", "ce_score": 0.01, "doc_path": "doc1.html"},
            ],
            "llm_context_prompt": "",
        }

        res = self.generator.generate("What is the capital of France?", stream=False)

        # Must be marked out-of-domain
        self.assertTrue(res["is_out_of_domain"])
        self.assertIn("outside the scope", res["answer"])
        # Crucially: LLM generate_content should NOT be called!
        self.mock_client.models.generate_content.assert_not_called()
        self.assertEqual(res["timings"]["generation_ms"], 0.0)

    def test_out_of_domain_short_circuit_local(self):
        # Top chunk has extremely negative logit from MiniLM
        self.mock_pipeline.query.return_value = {
            "query": "How to bake a chocolate cake?",
            "reranker_backend": "Local CPU/MPS (ms-marco-MiniLM-L-6-v2)",
            "timings": {"total_latency_ms": 40.0, "stage1_hybrid_ms": 25.0, "stage2_rerank_ms": 15.0},
            "chunks": [
                {"rank": 1, "chunk_id": "c1", "ce_score": -9.5, "doc_path": "doc1.html"},
            ],
            "llm_context_prompt": "",
        }

        res = self.generator.generate("How to bake a chocolate cake?", stream=False)

        self.assertTrue(res["is_out_of_domain"])
        self.assertIn("outside the scope", res["answer"])
        self.mock_client.models.generate_content.assert_not_called()

    def test_max_output_tokens_configured(self):
        self.assertEqual(self.generator.max_output_tokens, 4096)


if __name__ == "__main__":
    unittest.main()

