"""
tests/test_retrieval.py
-----------------------
Unit test suite for:
  - Code-aware tokenizer
  - BM25Retriever (indexing, scoring, serialization)
  - Information Retrieval evaluation metrics (Hit, Recall, Precision, MRR, MAP, nDCG)
"""

import math
import tempfile
import unittest
from pathlib import Path

from src.retrieval.bm25_retriever import BM25Retriever, tokenize_code_and_text
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
    convex_score_fusion,
    minmax_normalize,
)
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.metrics import (
    calculate_hit_at_k,
    calculate_recall_at_k,
    calculate_precision_at_k,
    calculate_mrr,
    calculate_ap,
    calculate_ndcg_at_k,
    evaluate_query,
    aggregate_metrics,
)


class TestBM25Tokenizer(unittest.TestCase):
    def test_dotted_identifiers_extracted(self):
        text = "How does itertools.batched() compare with os.path.join?"
        tokens = tokenize_code_and_text(text)
        self.assertIn("itertools.batched", tokens)
        self.assertIn("os.path.join", tokens)
        self.assertIn("itertools", tokens)
        self.assertIn("batched", tokens)
        self.assertIn("path", tokens)
        self.assertIn("join", tokens)

    def test_snake_case_subwords(self):
        text = "hashlib.file_digest reads binary files."
        tokens = tokenize_code_and_text(text)
        self.assertIn("hashlib.file_digest", tokens)
        self.assertIn("file_digest", tokens)
        self.assertIn("file", tokens)
        self.assertIn("digest", tokens)

    def test_case_insensitivity(self):
        t1 = tokenize_code_and_text("AsyncIO.TaskGroup")
        t2 = tokenize_code_and_text("asyncio.taskgroup")
        self.assertIn("asyncio.taskgroup", t1)
        self.assertIn("asyncio.taskgroup", t2)


class TestBM25Retriever(unittest.TestCase):
    def setUp(self):
        self.toy_chunks = [
            {
                "chunk_id": "chunk_asyncio_tg",
                "searchable_text": "Signature: asyncio.TaskGroup. An asynchronous context manager holding a group of tasks.",
                "relative_path": "library/asyncio.html",
                "doc_title": "asyncio",
            },
            {
                "chunk_id": "chunk_itertools_batched",
                "searchable_text": "Signature: itertools.batched(iterable, n). Batch data into tuples of length n.",
                "relative_path": "library/itertools.html",
                "doc_title": "itertools",
            },
            {
                "chunk_id": "chunk_os_path_join",
                "searchable_text": "Signature: os.path.join(path, *paths). Concatenate path segments with directory separators.",
                "relative_path": "library/os.path.html",
                "doc_title": "os.path",
            },
        ]
        self.retriever = BM25Retriever(k1=1.5, b=0.75)
        self.retriever.index_corpus(self.toy_chunks)

    def test_retrieval_exact_match(self):
        results = self.retriever.retrieve("itertools.batched batch tuples", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "chunk_itertools_batched")
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertEqual(results[0]["rank"], 1)

    def test_top_k_limits(self):
        results = self.retriever.retrieve("task or path", top_k=1)
        self.assertEqual(len(results), 1)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_bm25.pkl.gz"
            self.retriever.save(save_path)
            self.assertTrue(save_path.exists())

            loaded = BM25Retriever.load(save_path)
            self.assertTrue(loaded.is_indexed)
            self.assertEqual(len(loaded.chunk_ids), 3)

            results_orig = self.retriever.retrieve("asyncio TaskGroup", top_k=2)
            results_loaded = loaded.retrieve("asyncio TaskGroup", top_k=2)
            self.assertEqual(results_orig[0]["chunk_id"], results_loaded[0]["chunk_id"])
            self.assertEqual(results_orig[0]["score"], results_loaded[0]["score"])


class TestIRMetrics(unittest.TestCase):
    def test_hit_at_k(self):
        retrieved = ["c1", "c2", "c3", "c4"]
        target = {"c2", "c9"}
        self.assertEqual(calculate_hit_at_k(retrieved, target, k=1), 0)
        self.assertEqual(calculate_hit_at_k(retrieved, target, k=2), 1)
        self.assertEqual(calculate_hit_at_k(retrieved, target, k=5), 1)

    def test_recall_at_k(self):
        retrieved = ["c1", "c2", "c3", "c4"]
        target = {"c2", "c3"}
        self.assertAlmostEqual(calculate_recall_at_k(retrieved, target, k=1), 0.0)
        self.assertAlmostEqual(calculate_recall_at_k(retrieved, target, k=2), 0.5)
        self.assertAlmostEqual(calculate_recall_at_k(retrieved, target, k=3), 1.0)
        # Empty target check
        self.assertAlmostEqual(calculate_recall_at_k(retrieved, set(), k=3), 0.0)

    def test_precision_at_k(self):
        retrieved = ["c1", "c2", "c3", "c4"]
        target = {"c1", "c3"}
        self.assertAlmostEqual(calculate_precision_at_k(retrieved, target, k=1), 1.0)
        self.assertAlmostEqual(calculate_precision_at_k(retrieved, target, k=2), 0.5)
        self.assertAlmostEqual(calculate_precision_at_k(retrieved, target, k=4), 0.5)

    def test_mrr(self):
        # Target at rank 1
        self.assertAlmostEqual(calculate_mrr(["c1", "c2", "c3"], {"c1"}), 1.0)
        # Target at rank 2
        self.assertAlmostEqual(calculate_mrr(["c1", "c2", "c3"], {"c2"}), 0.5)
        # Target at rank 3
        self.assertAlmostEqual(calculate_mrr(["c1", "c2", "c3"], {"c3"}), 1.0 / 3.0)
        # Target not retrieved
        self.assertAlmostEqual(calculate_mrr(["c1", "c2", "c3"], {"c9"}), 0.0)

    def test_ap_and_map(self):
        # Retrieved: [hit, miss, hit], target={c1, c3}
        # P@1 = 1/1 = 1.0 (hit)
        # P@2 = 1/2 (miss)
        # P@3 = 2/3 (hit)
        # AP = (1.0 + 2/3) / 2 = 1.6666... / 2 = 5/6 ~= 0.8333
        retrieved = ["c1", "c2", "c3"]
        target = {"c1", "c3"}
        self.assertAlmostEqual(calculate_ap(retrieved, target, k=3), 5.0 / 6.0)

    def test_ndcg_at_k(self):
        # Target relevance: c1: 3, c2: 2
        # Ideal order: [c1, c2]
        # IDCG@2 = (2^3 - 1)/log2(2) + (2^2 - 1)/log2(3) = 7/1 + 3/1.58496 = 7 + 1.8928 = 8.8928
        target_rel = {"c1": 3, "c2": 2}
        
        # Perfect retrieval: [c1, c2] -> nDCG = 1.0
        self.assertAlmostEqual(calculate_ndcg_at_k(["c1", "c2"], target_rel, k=2), 1.0)
        
        # Inverted retrieval: [c2, c1]
        # DCG@2 = (2^2 - 1)/1 + (2^3 - 1)/log2(3) = 3 + 7/1.58496 = 3 + 4.4165 = 7.4165
        idcg_2 = 7.0 + 3.0 / math.log2(3)
        dcg_2_inv = 3.0 + 7.0 / math.log2(3)
        expected_ndcg = dcg_2_inv / idcg_2
        self.assertAlmostEqual(calculate_ndcg_at_k(["c2", "c1"], target_rel, k=2), expected_ndcg)
        
        # Zero hit
        self.assertAlmostEqual(calculate_ndcg_at_k(["c9", "c8"], target_rel, k=2), 0.0)

    def test_evaluate_query_integration(self):
        q = {
            "query_id": "Q_test",
            "category": "exact_api_lookup",
            "relevant_chunk_ids": ["c1"],
            "relevant_chunks": [{"chunk_id": "c1", "relevance_score": 3}]
        }
        retrieved = ["c1", "c2", "c3", "c4", "c5"]
        metrics = evaluate_query(q, retrieved, k_values=[1, 3, 5])
        self.assertEqual(metrics["hit@1"], 1)
        self.assertEqual(metrics["recall@1"], 1.0)
        self.assertEqual(metrics["precision@1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        self.assertEqual(metrics["ndcg@5"], 1.0)


class TestDenseRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toy_chunks = [
            {
                "chunk_id": "chunk_asyncio_tg",
                "searchable_text": "Signature: asyncio.TaskGroup. An asynchronous context manager holding a group of tasks.",
                "relative_path": "library/asyncio.html",
                "doc_title": "asyncio",
            },
            {
                "chunk_id": "chunk_itertools_batched",
                "searchable_text": "Signature: itertools.batched(iterable, n). Batch data into tuples of length n.",
                "relative_path": "library/itertools.html",
                "doc_title": "itertools",
            },
            {
                "chunk_id": "chunk_os_path_join",
                "searchable_text": "Signature: os.path.join(path, *paths). Concatenate path segments with directory separators.",
                "relative_path": "library/os.path.html",
                "doc_title": "os.path",
            },
        ]
        # Use CPU for deterministic fast unit testing
        cls.retriever = DenseRetriever(model_name="BAAI/bge-small-en-v1.5", device="cpu")
        cls.retriever.index_corpus(cls.toy_chunks, verbose=False)

    def test_embeddings_shape_and_normalization(self):
        self.assertTrue(self.retriever.is_indexed)
        self.assertEqual(self.retriever.embeddings.shape, (3, 384))
        # Check unit L2 norm
        norms = (self.retriever.embeddings ** 2).sum(axis=1)
        for n in norms:
            self.assertAlmostEqual(n, 1.0, places=4)

    def test_dense_retrieval(self):
        results = self.retriever.retrieve("group of asynchronous coroutines and tasks", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["chunk_id"], "chunk_asyncio_tg")
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertEqual(results[0]["rank"], 1)

    def test_dense_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "dense_test_index"
            self.retriever.save(save_path)
            self.assertTrue(save_path.with_suffix(".meta.pkl.gz").exists())
            self.assertTrue(save_path.with_suffix(".embeddings.npz").exists())

            loaded = DenseRetriever.load(save_path, device="cpu")
            self.assertTrue(loaded.is_indexed)
            self.assertEqual(loaded.embeddings.shape, (3, 384))

            orig_res = self.retriever.retrieve("batch data into chunks", top_k=2)
            load_res = loaded.retrieve("batch data into chunks", top_k=2)
            self.assertEqual(orig_res[0]["chunk_id"], load_res[0]["chunk_id"])
            self.assertAlmostEqual(orig_res[0]["score"], load_res[0]["score"], places=4)


class TestHybridRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.toy_chunks = [
            {
                "chunk_id": "chunk_asyncio_tg",
                "searchable_text": "Signature: asyncio.TaskGroup. An asynchronous context manager holding a group of tasks.",
                "relative_path": "library/asyncio.html",
                "doc_title": "asyncio",
            },
            {
                "chunk_id": "chunk_itertools_batched",
                "searchable_text": "Signature: itertools.batched(iterable, n). Batch data into tuples of length n.",
                "relative_path": "library/itertools.html",
                "doc_title": "itertools",
            },
            {
                "chunk_id": "chunk_os_path_join",
                "searchable_text": "Signature: os.path.join(path, *paths). Concatenate path segments with directory separators.",
                "relative_path": "library/os.path.html",
                "doc_title": "os.path",
            },
        ]
        cls.bm25 = BM25Retriever(k1=1.5, b=0.75)
        cls.bm25.index_corpus(cls.toy_chunks)

        cls.dense = DenseRetriever(model_name="BAAI/bge-small-en-v1.5", device="cpu")
        cls.dense.index_corpus(cls.toy_chunks, verbose=False)

        cls.hybrid = HybridRetriever(bm25_retriever=cls.bm25, dense_retriever=cls.dense)

    def test_rrf_basic_calculation(self):
        list1 = [{"chunk_id": "c1", "rank": 1, "score": 10.0}, {"chunk_id": "c2", "rank": 2, "score": 5.0}]
        list2 = [{"chunk_id": "c2", "rank": 1, "score": 0.9}, {"chunk_id": "c3", "rank": 2, "score": 0.8}]
        ranked_lists = [(list1, 1.0), (list2, 1.0)]

        results = reciprocal_rank_fusion(ranked_lists, k_rrf=60, top_k=10)
        self.assertEqual(len(results), 3)

        # Expected scores with k=60:
        # c1: 1/(60+1) = 1/61 ~= 0.0163934
        # c2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ~= 0.0161290 + 0.0163934 = 0.0325224
        # c3: 1/(60+2) = 1/62 ~= 0.0161290
        self.assertEqual(results[0]["chunk_id"], "c2")
        self.assertEqual(results[1]["chunk_id"], "c1")
        self.assertEqual(results[2]["chunk_id"], "c3")

        self.assertAlmostEqual(results[0]["score"], 1.0 / 62 + 1.0 / 61, places=5)
        self.assertAlmostEqual(results[1]["score"], 1.0 / 61, places=5)
        self.assertAlmostEqual(results[2]["score"], 1.0 / 62, places=5)

    def test_rrf_weighting(self):
        list1 = [{"chunk_id": "c1", "rank": 1}]
        list2 = [{"chunk_id": "c2", "rank": 1}]
        # Give method 1 double the weight
        ranked_lists = [(list1, 2.0), (list2, 1.0)]

        results = reciprocal_rank_fusion(ranked_lists, k_rrf=60, top_k=10)
        self.assertEqual(results[0]["chunk_id"], "c1")
        self.assertEqual(results[1]["chunk_id"], "c2")
        self.assertAlmostEqual(results[0]["score"], 2.0 / 61, places=5)
        self.assertAlmostEqual(results[1]["score"], 1.0 / 61, places=5)

    def test_minmax_normalize(self):
        # Regular scaling
        norm = minmax_normalize([10.0, 20.0, 30.0])
        self.assertAlmostEqual(norm[0], 0.0)
        self.assertAlmostEqual(norm[1], 0.5)
        self.assertAlmostEqual(norm[2], 1.0)

        # Equal values
        uniform = minmax_normalize([5.0, 5.0, 5.0])
        self.assertEqual(uniform, [1.0, 1.0, 1.0])

        # Empty
        self.assertEqual(minmax_normalize([]), [])

    def test_convex_score_fusion(self):
        bm25_cands = [
            {"chunk_id": "c1", "rank": 1, "score": 20.0},
            {"chunk_id": "c2", "rank": 2, "score": 10.0},
        ]
        dense_cands = [
            {"chunk_id": "c2", "rank": 1, "score": 0.8},
            {"chunk_id": "c3", "rank": 2, "score": 0.4},
        ]

        # Normalized bm25: c1=1.0, c2=0.0
        # Normalized dense: c2=1.0, c3=0.0
        # For alpha = 0.5:
        # c1: 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        # c2: 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        # c3: 0.5 * 0.0 + 0.5 * 0.0 = 0.0
        results = convex_score_fusion(bm25_cands, dense_cands, alpha=0.5, top_k=10)
        self.assertEqual(len(results), 3)

        score_lookup = {r["chunk_id"]: r["score"] for r in results}
        self.assertAlmostEqual(score_lookup["c1"], 0.5, places=5)
        self.assertAlmostEqual(score_lookup["c2"], 0.5, places=5)
        self.assertAlmostEqual(score_lookup["c3"], 0.0, places=5)

        # Pure dense: alpha = 1.0
        res_dense = convex_score_fusion(bm25_cands, dense_cands, alpha=1.0, top_k=10)
        dense_lookup = {r["chunk_id"]: r["score"] for r in res_dense}
        self.assertAlmostEqual(dense_lookup["c2"], 1.0, places=5)
        self.assertAlmostEqual(dense_lookup["c1"], 0.0, places=5)
        self.assertAlmostEqual(dense_lookup["c3"], 0.0, places=5)

    def test_hybrid_retriever_e2e(self):
        # Test RRF retrieval
        query = "asyncio TaskGroup async context manager"
        res_rrf = self.hybrid.retrieve(query, top_k=2, method="rrf", pool_size=3)
        self.assertEqual(len(res_rrf), 2)
        self.assertEqual(res_rrf[0]["chunk_id"], "chunk_asyncio_tg")
        self.assertEqual(res_rrf[0]["rank"], 1)
        self.assertIn("bm25_rank", res_rrf[0])
        self.assertIn("dense_rank", res_rrf[0])

        # Test Linear combination retrieval
        res_linear = self.hybrid.retrieve(query, top_k=2, method="linear", alpha=0.6, pool_size=3)
        self.assertEqual(len(res_linear), 2)
        self.assertEqual(res_linear[0]["chunk_id"], "chunk_asyncio_tg")
        self.assertEqual(res_linear[0]["rank"], 1)

        # Test batch retrieve
        queries = [
            {"query_id": "q1", "question": "asyncio TaskGroup"},
            {"query_id": "q2", "question": "itertools batch"},
        ]
        batch_res = self.hybrid.batch_retrieve(queries, top_k=2, method="rrf")
        self.assertEqual(len(batch_res), 2)
        self.assertEqual(batch_res["q1"][0]["chunk_id"], "chunk_asyncio_tg")
        self.assertEqual(batch_res["q2"][0]["chunk_id"], "chunk_itertools_batched")

        # Test invalid method exception
        with self.assertRaises(ValueError):
            self.hybrid.retrieve(query, method="invalid_method")


class TestCrossEncoderReranker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use CPU for deterministic fast unit test
        cls.reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            batch_size=8,
        )
        cls.toy_chunks = {
            "c_asyncio": {
                "chunk_id": "c_asyncio",
                "searchable_text": "asyncio.TaskGroup manages concurrent tasks and handles exceptions gracefully.",
                "preview": "asyncio.TaskGroup manages concurrent tasks",
            },
            "c_recipes": {
                "chunk_id": "c_recipes",
                "searchable_text": "Itertools recipes include pairwise, batched, and flatten iterator helpers.",
                "preview": "Itertools recipes include pairwise, batched",
            },
            "c_os_path": {
                "chunk_id": "c_os_path",
                "searchable_text": "os.path.join concatenates paths using system directory separators.",
                "preview": "os.path.join concatenates paths",
            },
        }

    def test_rerank_single_query(self):
        query = "How to run multiple asynchronous tasks with TaskGroup?"
        candidates = [
            {"chunk_id": "c_os_path", "rank": 1, "score": 0.8},
            {"chunk_id": "c_asyncio", "rank": 2, "score": 0.5},
            {"chunk_id": "c_recipes", "rank": 3, "score": 0.3},
        ]
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            chunk_lookup=self.toy_chunks,
            top_k=2,
        )
        self.assertEqual(len(reranked), 2)
        # c_asyncio should now be rank 1
        self.assertEqual(reranked[0]["chunk_id"], "c_asyncio")
        self.assertEqual(reranked[0]["rank"], 1)
        self.assertEqual(reranked[0]["original_rank"], 2)
        self.assertGreater(reranked[0]["score"], reranked[1]["score"])

    def test_batch_rerank(self):
        queries = [
            {"query_id": "q1", "question": "concurrent task group in asyncio"},
            {"query_id": "q2", "question": "path concatenation with separators"},
        ]
        candidates = {
            "q1": [
                {"chunk_id": "c_os_path", "rank": 1, "score": 0.9},
                {"chunk_id": "c_asyncio", "rank": 2, "score": 0.4},
            ],
            "q2": [
                {"chunk_id": "c_recipes", "rank": 1, "score": 0.7},
                {"chunk_id": "c_os_path", "rank": 2, "score": 0.5},
            ],
        }
        batch_results = self.reranker.batch_rerank(
            queries=queries,
            candidate_dict=candidates,
            chunk_lookup=self.toy_chunks,
            top_k=2,
        )
        self.assertEqual(len(batch_results), 2)
        self.assertEqual(batch_results["q1"][0]["chunk_id"], "c_asyncio")
        self.assertEqual(batch_results["q2"][0]["chunk_id"], "c_os_path")

    def test_empty_candidates(self):
        res = self.reranker.rerank("some query", [])
        self.assertEqual(res, [])



if __name__ == "__main__":
    unittest.main()

