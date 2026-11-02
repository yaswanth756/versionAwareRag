"""
src/retrieval/metrics.py
------------------------
Information Retrieval (IR) evaluation metrics module for RAG benchmarking:
  - Hit@K
  - Recall@K
  - Precision@K
  - MRR (Mean Reciprocal Rank)
  - MAP / AP@K (Average Precision)
  - nDCG@K (Normalized Discounted Cumulative Gain with graded relevance)
"""

import math
from typing import List, Set, Dict, Any, Optional


def calculate_hit_at_k(retrieved_cids: List[str], target_cids: Set[str], k: int) -> int:
    """Returns 1 if at least one target chunk is in top-k retrieved chunks, else 0."""
    top_k = retrieved_cids[:k]
    return 1 if any(cid in target_cids for cid in top_k) else 0


def calculate_recall_at_k(retrieved_cids: List[str], target_cids: Set[str], k: int) -> float:
    """
    Computes Recall@k: |Retrieved_k ∩ Target| / |Target|.
    Returns 0.0 if target_cids is empty.
    """
    if not target_cids:
        return 0.0
    top_k = set(retrieved_cids[:k])
    hits = len(top_k.intersection(target_cids))
    return hits / float(len(target_cids))


def calculate_precision_at_k(retrieved_cids: List[str], target_cids: Set[str], k: int) -> float:
    """
    Computes Precision@k: |Retrieved_k ∩ Target| / k.
    """
    if k <= 0:
        return 0.0
    top_k = set(retrieved_cids[:k])
    hits = len(top_k.intersection(target_cids))
    return hits / float(k)


def calculate_mrr(retrieved_cids: List[str], target_cids: Set[str], max_k: Optional[int] = None) -> float:
    """
    Computes Reciprocal Rank (RR): 1 / rank of first relevant chunk.
    If no relevant chunk is retrieved within max_k (or all retrieved), returns 0.0.
    """
    candidates = retrieved_cids if max_k is None else retrieved_cids[:max_k]
    for rank, cid in enumerate(candidates, start=1):
        if cid in target_cids:
            return 1.0 / rank
    return 0.0


def calculate_ap(retrieved_cids: List[str], target_cids: Set[str], k: int = 20) -> float:
    """
    Computes Average Precision (AP@k) for a single query.
    AP@k = (1 / |Target|) * sum_{i=1}^k (Precision@i * Indicator(chunk_i in Target))
    """
    if not target_cids or k <= 0:
        return 0.0

    candidates = retrieved_cids[:k]
    hits = 0
    prec_sum = 0.0

    for i, cid in enumerate(candidates, start=1):
        if cid in target_cids:
            hits += 1
            prec_sum += hits / float(i)

    # Standard TREC evaluation normalizes by total number of ground truth relevant documents
    return prec_sum / float(len(target_cids))


def calculate_ndcg_at_k(
    retrieved_cids: List[str],
    target_relevance: Dict[str, int],
    k: int = 10
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain (nDCG@k) using graded relevance.
    DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    IDCG@k = DCG@k of ideally sorted ground-truth relevant chunks.
    """
    if not target_relevance or k <= 0:
        return 0.0

    # 1. Compute DCG@k
    dcg = 0.0
    candidates = retrieved_cids[:k]
    for i, cid in enumerate(candidates, start=1):
        rel = target_relevance.get(cid, 0)
        if rel > 0:
            dcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

    # 2. Compute IDCG@k
    ideal_scores = sorted(target_relevance.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(ideal_scores, start=1):
        if rel > 0:
            idcg += (math.pow(2, rel) - 1.0) / math.log2(i + 1)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


def evaluate_query(
    query_record: Dict[str, Any],
    retrieved_cids: List[str],
    k_values: Optional[List[int]] = None,
    chunk_to_doc_id: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Evaluates retrieval for a single query against all standard metrics.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 20]

    target_cids = set(query_record.get("relevant_chunk_ids", []))
    target_relevance = {
        rc["chunk_id"]: rc.get("relevance_score", 3)
        for rc in query_record.get("relevant_chunks", [])
    }

    metrics: Dict[str, Any] = {
        "query_id": query_record.get("query_id", ""),
        "category": query_record.get("category", ""),
        "num_relevant_chunks": len(target_cids),
        "mrr": calculate_mrr(retrieved_cids, target_cids),
        "map_at_20": calculate_ap(retrieved_cids, target_cids, k=20),
    }

    # Cutoff metrics
    for k in k_values:
        metrics[f"hit@{k}"] = calculate_hit_at_k(retrieved_cids, target_cids, k)
        metrics[f"recall@{k}"] = calculate_recall_at_k(retrieved_cids, target_cids, k)
        metrics[f"precision@{k}"] = calculate_precision_at_k(retrieved_cids, target_cids, k)
        if k in [5, 10, 20]:
            metrics[f"ndcg@{k}"] = calculate_ndcg_at_k(retrieved_cids, target_relevance, k)

    # Document-level evaluation (if mapping provided)
    if chunk_to_doc_id:
        target_docs = {chunk_to_doc_id.get(cid) for cid in target_cids if cid in chunk_to_doc_id}
        target_docs.discard(None)
        if target_docs:
            for k in [1, 5, 10]:
                retrieved_docs = [chunk_to_doc_id.get(cid) for cid in retrieved_cids[:k]]
                metrics[f"doc_hit@{k}"] = 1 if any(doc in target_docs for doc in retrieved_docs) else 0

    return metrics


def aggregate_metrics(query_evals: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates per-query evaluations into mean metric scores.
    """
    if not query_evals:
        return {}

    metric_keys = [k for k, v in query_evals[0].items() if isinstance(v, (int, float)) and k != "num_relevant_chunks"]
    aggregates = {k: 0.0 for k in metric_keys}

    for qe in query_evals:
        for k in metric_keys:
            aggregates[k] += qe.get(k, 0.0)

    n = float(len(query_evals))
    return {k: round(v / n, 4) for k, v in aggregates.items()}

