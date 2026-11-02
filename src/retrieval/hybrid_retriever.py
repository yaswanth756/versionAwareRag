"""
src/retrieval/hybrid_retriever.py
---------------------------------
Hybrid Information Retrieval combining lexical BM25 and dense semantic embeddings.
Provides:
  - Reciprocal Rank Fusion (RRF):
      RRF_score(d) = sum_{m} w_m / (k_rrf + rank_m(d))
  - Convex Score Normalization (Linear Combination):
      score(d) = alpha * norm(score_dense) + (1 - alpha) * norm(score_bm25)
  - Modular HybridRetriever class with single-query and batch retrieval interfaces.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever


def reciprocal_rank_fusion(
    ranked_lists: List[Tuple[List[Dict[str, Any]], float]],
    k_rrf: int = 60,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Performs Reciprocal Rank Fusion (RRF) over multiple ranked candidate lists.

    Args:
        ranked_lists: List of tuples `(candidate_list, weight)`, where candidate_list is a
                      list of dicts containing at least `chunk_id` and `rank` (1-based),
                      and weight is the method weight multiplier.
        k_rrf: Smoothing constant to penalize low ranks without discarding them (default: 60).
        top_k: Maximum number of fused candidates to return.

    Returns:
        List of fused candidate dicts sorted by descending RRF score.
    """
    if not ranked_lists:
        return []

    fused_scores: Dict[str, float] = {}
    chunk_metadata: Dict[str, Dict[str, Any]] = {}
    method_ranks: Dict[str, Dict[int, int]] = {}
    method_scores: Dict[str, Dict[int, float]] = {}

    for method_idx, (candidates, weight) in enumerate(ranked_lists):
        for item in candidates:
            cid = item["chunk_id"]
            rank = item["rank"]
            score = item.get("score", 0.0)

            if cid not in fused_scores:
                fused_scores[cid] = 0.0
                chunk_metadata[cid] = {
                    "chunk_id": cid,
                    "doc_path": item.get("doc_path", ""),
                    "doc_title": item.get("doc_title", ""),
                    "breadcrumbs": item.get("breadcrumbs", []),
                    "preview": item.get("preview", ""),
                }
                method_ranks[cid] = {}
                method_scores[cid] = {}

            fused_scores[cid] += weight * (1.0 / (k_rrf + rank))
            method_ranks[cid][method_idx] = rank
            method_scores[cid][method_idx] = score

    sorted_cids = sorted(fused_scores.keys(), key=lambda c: fused_scores[c], reverse=True)[:top_k]

    results = []
    for rank, cid in enumerate(sorted_cids, start=1):
        meta = chunk_metadata[cid]
        entry = {
            "rank": rank,
            "chunk_id": cid,
            "score": round(float(fused_scores[cid]), 6),
            "doc_path": meta["doc_path"],
            "doc_title": meta["doc_title"],
            "breadcrumbs": meta["breadcrumbs"],
            "preview": meta["preview"],
            "method_ranks": method_ranks[cid],
            "method_scores": method_scores[cid],
        }
        results.append(entry)

    return results


def minmax_normalize(scores: List[float], eps: float = 1e-8) -> List[float]:
    """
    Applies Min-Max scaling to a list of scores: (s - min) / (max - min + eps).
    If all scores are equal, returns [1.0, ...].
    """
    if not scores:
        return []
    s_min = min(scores)
    s_max = max(scores)
    diff = s_max - s_min
    if abs(diff) < eps:
        return [1.0] * len(scores)
    return [(s - s_min) / (diff + eps) for s in scores]


def convex_score_fusion(
    bm25_candidates: List[Dict[str, Any]],
    dense_candidates: List[Dict[str, Any]],
    alpha: float = 0.5,
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Performs convex combination of normalized BM25 and Dense scores:
        score(d) = alpha * norm(dense_score) + (1 - alpha) * norm(bm25_score)

    Missing candidates receive 0.0 normalized score for that branch.
    """
    bm25_raw = [c.get("score", 0.0) for c in bm25_candidates]
    dense_raw = [c.get("score", 0.0) for c in dense_candidates]

    bm25_norm = minmax_normalize(bm25_raw)
    dense_norm = minmax_normalize(dense_raw)

    chunk_data: Dict[str, Dict[str, Any]] = {}

    for c, norm_s in zip(bm25_candidates, bm25_norm):
        cid = c["chunk_id"]
        chunk_data[cid] = {
            "chunk_id": cid,
            "doc_path": c.get("doc_path", ""),
            "doc_title": c.get("doc_title", ""),
            "breadcrumbs": c.get("breadcrumbs", []),
            "preview": c.get("preview", ""),
            "bm25_norm": norm_s,
            "dense_norm": 0.0,
            "bm25_rank": c.get("rank"),
            "dense_rank": None,
            "bm25_score": c.get("score", 0.0),
            "dense_score": None,
        }

    for c, norm_s in zip(dense_candidates, dense_norm):
        cid = c["chunk_id"]
        if cid not in chunk_data:
            chunk_data[cid] = {
                "chunk_id": cid,
                "doc_path": c.get("doc_path", ""),
                "doc_title": c.get("doc_title", ""),
                "breadcrumbs": c.get("breadcrumbs", []),
                "preview": c.get("preview", ""),
                "bm25_norm": 0.0,
                "dense_norm": norm_s,
                "bm25_rank": None,
                "dense_rank": c.get("rank"),
                "bm25_score": None,
                "dense_score": c.get("score", 0.0),
            }
        else:
            chunk_data[cid]["dense_norm"] = norm_s
            chunk_data[cid]["dense_rank"] = c.get("rank")
            chunk_data[cid]["dense_score"] = c.get("score", 0.0)

    for cid, data in chunk_data.items():
        data["fused_score"] = alpha * data["dense_norm"] + (1.0 - alpha) * data["bm25_norm"]

    sorted_items = sorted(chunk_data.values(), key=lambda x: x["fused_score"], reverse=True)[:top_k]

    results = []
    for rank, item in enumerate(sorted_items, start=1):
        results.append({
            "rank": rank,
            "chunk_id": item["chunk_id"],
            "score": round(float(item["fused_score"]), 6),
            "doc_path": item["doc_path"],
            "doc_title": item["doc_title"],
            "breadcrumbs": item["breadcrumbs"],
            "preview": item["preview"],
            "bm25_rank": item["bm25_rank"],
            "dense_rank": item["dense_rank"],
            "bm25_score": item["bm25_score"],
            "dense_score": item["dense_score"],
            "bm25_norm": round(float(item["bm25_norm"]), 6),
            "dense_norm": round(float(item["dense_norm"]), 6),
        })

    return results


class HybridRetriever:
    """
    Hybrid Retriever integrating BM25 lexical and Dense vector semantic retrieval.
    """

    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        dense_retriever: DenseRetriever,
    ):
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever

    @classmethod
    def load(
        cls,
        bm25_index_path: Union[str, Path],
        dense_index_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> "HybridRetriever":
        """
        Loads pre-computed BM25 and Dense indices from disk.
        """
        bm25 = BM25Retriever.load(bm25_index_path)
        dense = DenseRetriever.load(dense_index_path, device=device)
        return cls(bm25_retriever=bm25, dense_retriever=dense)

    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        method: str = "rrf",
        pool_size: int = 50,
        k_rrf: int = 60,
        weight_bm25: float = 1.0,
        weight_dense: float = 1.0,
        alpha: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Performs hybrid retrieval for a single query.

        Args:
            query: Query text string.
            top_k: Number of fused results to return.
            method: 'rrf' (Reciprocal Rank Fusion) or 'linear' (Convex combination).
            pool_size: Number of candidate chunks retrieved from each single retriever.
            k_rrf: RRF smoothing constant (used when method='rrf').
            weight_bm25: Weight multiplier for BM25 branch in RRF.
            weight_dense: Weight multiplier for Dense branch in RRF.
            alpha: Weight of dense model in linear combination (method='linear', 0 <= alpha <= 1).

        Returns:
            List of top_k candidate chunks with fused ranks and scores.
        """
        bm25_res = self.bm25_retriever.retrieve(query, top_k=pool_size)
        dense_res = self.dense_retriever.retrieve(query, top_k=pool_size)

        if method == "rrf":
            ranked_lists = [
                (bm25_res, weight_bm25),
                (dense_res, weight_dense),
            ]
            raw_fused = reciprocal_rank_fusion(ranked_lists, k_rrf=k_rrf, top_k=top_k)
            # Annotate with branch-specific rank information
            formatted = []
            for item in raw_fused:
                m_ranks = item.get("method_ranks", {})
                m_scores = item.get("method_scores", {})
                formatted.append({
                    "rank": item["rank"],
                    "chunk_id": item["chunk_id"],
                    "score": item["score"],
                    "doc_path": item["doc_path"],
                    "doc_title": item["doc_title"],
                    "breadcrumbs": item["breadcrumbs"],
                    "preview": item["preview"],
                    "bm25_rank": m_ranks.get(0),
                    "dense_rank": m_ranks.get(1),
                    "bm25_score": m_scores.get(0),
                    "dense_score": m_scores.get(1),
                })
            return formatted

        elif method == "linear":
            return convex_score_fusion(
                bm25_candidates=bm25_res,
                dense_candidates=dense_res,
                alpha=alpha,
                top_k=top_k,
            )
        else:
            raise ValueError(f"Unknown fusion method '{method}'. Supported: 'rrf', 'linear'.")

    def batch_retrieve(
        self,
        queries: List[Dict[str, Any]],
        top_k: int = 20,
        method: str = "rrf",
        pool_size: int = 50,
        k_rrf: int = 60,
        weight_bm25: float = 1.0,
        weight_dense: float = 1.0,
        alpha: float = 0.5,
        query_text_key: str = "question",
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch-retrieves candidates for multiple queries.
        Optimizes dense retrieval using batch vector encoding.
        """
        # Batch retrieve from BM25 and Dense
        bm25_batch = self.bm25_retriever.batch_retrieve(
            queries, top_k=pool_size, query_text_key=query_text_key
        )
        dense_batch = self.dense_retriever.batch_retrieve(
            queries, top_k=pool_size, query_text_key=query_text_key
        )

        results: Dict[str, List[Dict[str, Any]]] = {}

        for q in queries:
            qid = q["query_id"]
            bm25_res = bm25_batch.get(qid, [])
            dense_res = dense_batch.get(qid, [])

            if method == "rrf":
                ranked_lists = [
                    (bm25_res, weight_bm25),
                    (dense_res, weight_dense),
                ]
                raw_fused = reciprocal_rank_fusion(ranked_lists, k_rrf=k_rrf, top_k=top_k)
                formatted = []
                for item in raw_fused:
                    m_ranks = item.get("method_ranks", {})
                    m_scores = item.get("method_scores", {})
                    formatted.append({
                        "rank": item["rank"],
                        "chunk_id": item["chunk_id"],
                        "score": item["score"],
                        "doc_path": item["doc_path"],
                        "doc_title": item["doc_title"],
                        "breadcrumbs": item["breadcrumbs"],
                        "preview": item["preview"],
                        "bm25_rank": m_ranks.get(0),
                        "dense_rank": m_ranks.get(1),
                        "bm25_score": m_scores.get(0),
                        "dense_score": m_scores.get(1),
                    })
                results[qid] = formatted
            elif method == "linear":
                results[qid] = convex_score_fusion(
                    bm25_candidates=bm25_res,
                    dense_candidates=dense_res,
                    alpha=alpha,
                    top_k=top_k,
                )
            else:
                raise ValueError(f"Unknown fusion method '{method}'.")

        return results

