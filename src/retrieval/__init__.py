"""
src/retrieval
-------------
Information Retrieval package for the Python 3.12 documentation RAG pipeline.
Provides:
  - BM25Retriever: Lexical BM25 retrieval over documentation chunks.
  - DenseRetriever: Semantic dense vector retrieval using SentenceTransformers.
  - HybridRetriever: Hybrid retrieval combining BM25 and Dense with RRF / convex score fusion.
  - CrossEncoderReranker: Neural cross-encoder reranker refining top candidate chunk rankings.
  - Metrics: Comprehensive Information Retrieval evaluation metrics (Recall, Precision, MRR, MAP, nDCG).
"""

from src.retrieval.bm25_retriever import BM25Retriever, tokenize_code_and_text
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid_retriever import (
    HybridRetriever,
    reciprocal_rank_fusion,
    convex_score_fusion,
    minmax_normalize,
)
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.remote_reranker import RemoteCrossEncoderReranker
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

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "CrossEncoderReranker",
    "RemoteCrossEncoderReranker",
    "reciprocal_rank_fusion",
    "convex_score_fusion",
    "minmax_normalize",
    "tokenize_code_and_text",
    "calculate_hit_at_k",
    "calculate_recall_at_k",
    "calculate_precision_at_k",
    "calculate_mrr",
    "calculate_ap",
    "calculate_ndcg_at_k",
    "evaluate_query",
    "aggregate_metrics",
]
