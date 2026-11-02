"""
src/retrieval/reranker.py
-------------------------
Cross-Encoder Neural Reranking module for the Python 3.12 documentation RAG pipeline.
Performs full cross-attention over (query, document_chunk) pairs to refine candidate
rankings retrieved by first-stage retrievers (BM25, Dense, or Hybrid fusion).
Supports pure cross-encoder scoring and blended score interpolation (Cross-Encoder + First-Stage Prior).
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import numpy as np

# Ensure offline operation if models are cached locally
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from sentence_transformers import CrossEncoder

from src.retrieval.hybrid_retriever import minmax_normalize


def get_default_device() -> str:
    """Returns 'mps' on Apple Silicon, 'cuda' if available, otherwise 'cpu'."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CrossEncoderReranker:
    """
    Cross-Encoder neural reranker using full cross-attention over query-document pairs.
    Scores (query, text) joint embeddings to produce refined relevance scores.
    Supports pure reranking (blend_beta=1.0) and score blending with first-stage prior (e.g. blend_beta=0.7).
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = device or get_default_device()
        self.max_length = max_length
        self.batch_size = batch_size
        self.model: Optional[CrossEncoder] = None

    def _load_model(self) -> CrossEncoder:
        if self.model is None:
            try:
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                    local_files_only=True,
                )
            except Exception:
                self.model = CrossEncoder(
                    self.model_name,
                    max_length=self.max_length,
                    device=self.device,
                )
        return self.model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        chunk_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        text_field: str = "searchable_text",
        top_k: Optional[int] = None,
        blend_beta: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunks for a single query.

        Args:
            query: Question or query string.
            candidates: List of candidate chunk dictionaries (from first-stage retrieval).
            chunk_lookup: Optional dict mapping chunk_id to full chunk record.
            text_field: Field name containing text to feed into the cross-encoder.
            top_k: Optional limit on number of reranked results returned.
            blend_beta: Weight of Cross-Encoder in score interpolation (0.0 to 1.0).
                        1.0 = Pure Cross-Encoder.
                        0.7 = 70% Cross-Encoder + 30% First-Stage retrieval score.

        Returns:
            List of reranked candidate dicts sorted in descending order of final score.
        """
        if not candidates:
            return []

        self._load_model()

        # Build (query, passage) pairs
        pairs = []
        for c in candidates:
            cid = c["chunk_id"]
            if chunk_lookup and cid in chunk_lookup:
                passage_text = chunk_lookup[cid].get(text_field, c.get("preview", ""))
            elif text_field in c:
                passage_text = c[text_field]
            else:
                passage_text = c.get("preview", "")
            pairs.append((query, passage_text))

        # Compute cross-encoder scores
        raw_ce_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        raw_ce_scores = np.asarray(raw_ce_scores, dtype=np.float32).reshape(-1)

        # Apply score blending if blend_beta < 1.0
        if blend_beta < 1.0 and any("score" in c for c in candidates):
            ce_norm = np.array(minmax_normalize(raw_ce_scores.tolist()), dtype=np.float32)
            orig_raw = [float(c.get("score", 0.0)) for c in candidates]
            orig_norm = np.array(minmax_normalize(orig_raw), dtype=np.float32)
            final_scores = blend_beta * ce_norm + (1.0 - blend_beta) * orig_norm
        else:
            final_scores = raw_ce_scores

        # Sort candidate indices by descending final score
        ranked_indices = sorted(range(len(candidates)), key=lambda i: final_scores[i], reverse=True)
        if top_k is not None:
            ranked_indices = ranked_indices[:top_k]

        reranked_results = []
        for rank, idx in enumerate(ranked_indices, start=1):
            cand = candidates[idx]
            entry = {
                "rank": rank,
                "chunk_id": cand["chunk_id"],
                "score": round(float(final_scores[idx]), 6),
                "ce_score": round(float(raw_ce_scores[idx]), 6),
                "original_rank": cand.get("rank"),
                "original_score": cand.get("score"),
                "doc_path": cand.get("doc_path", ""),
                "doc_title": cand.get("doc_title", ""),
                "breadcrumbs": cand.get("breadcrumbs", []),
                "preview": cand.get("preview", ""),
            }
            for key in ["bm25_rank", "dense_rank", "bm25_score", "dense_score"]:
                if key in cand:
                    entry[key] = cand[key]

            reranked_results.append(entry)

        return reranked_results

    def batch_rerank(
        self,
        queries: List[Dict[str, Any]],
        candidate_dict: Dict[str, List[Dict[str, Any]]],
        chunk_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        text_field: str = "searchable_text",
        top_k: Optional[int] = None,
        query_text_key: str = "question",
        blend_beta: float = 1.0,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Batch-reranks candidates across multiple queries in a unified tensor pass.

        Args:
            queries: List of query dicts (must have 'query_id' and query_text_key).
            candidate_dict: Mapping of query_id to list of candidate chunk dicts.
            chunk_lookup: Mapping of chunk_id to full chunk record.
            text_field: Field name for passage text.
            top_k: Number of reranked candidates to retain per query.
            query_text_key: Field name for query text.
            blend_beta: Weight of Cross-Encoder in score interpolation (0.0 to 1.0).

        Returns:
            Dict mapping query_id to list of reranked candidate dicts.
        """
        self._load_model()

        # Flatten all pairs across all queries
        all_pairs = []
        query_slices = []  # (qid, start_idx, end_idx, candidates)

        for q in queries:
            qid = q["query_id"]
            q_text = q[query_text_key]
            cands = candidate_dict.get(qid, [])
            start_idx = len(all_pairs)

            for c in cands:
                cid = c["chunk_id"]
                if chunk_lookup and cid in chunk_lookup:
                    passage_text = chunk_lookup[cid].get(text_field, c.get("preview", ""))
                elif text_field in c:
                    passage_text = c[text_field]
                else:
                    passage_text = c.get("preview", "")
                all_pairs.append((q_text, passage_text))

            end_idx = len(all_pairs)
            query_slices.append((qid, start_idx, end_idx, cands))

        if not all_pairs:
            return {q["query_id"]: [] for q in queries}

        # Predict all pairs in batched mode on device
        all_ce_scores = self.model.predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        all_ce_scores = np.asarray(all_ce_scores, dtype=np.float32).reshape(-1)

        # Slice scores back to queries and sort
        results: Dict[str, List[Dict[str, Any]]] = {}

        for qid, start_idx, end_idx, cands in query_slices:
            if start_idx == end_idx:
                results[qid] = []
                continue

            q_ce_scores = all_ce_scores[start_idx:end_idx]

            # Apply score blending if blend_beta < 1.0
            if blend_beta < 1.0 and any("score" in c for c in cands):
                ce_norm = np.array(minmax_normalize(q_ce_scores.tolist()), dtype=np.float32)
                orig_raw = [float(c.get("score", 0.0)) for c in cands]
                orig_norm = np.array(minmax_normalize(orig_raw), dtype=np.float32)
                final_scores = blend_beta * ce_norm + (1.0 - blend_beta) * orig_norm
            else:
                final_scores = q_ce_scores

            ranked_indices = sorted(range(len(cands)), key=lambda i: final_scores[i], reverse=True)
            if top_k is not None:
                ranked_indices = ranked_indices[:top_k]

            reranked = []
            for rank, idx in enumerate(ranked_indices, start=1):
                cand = cands[idx]
                entry = {
                    "rank": rank,
                    "chunk_id": cand["chunk_id"],
                    "score": round(float(final_scores[idx]), 6),
                    "ce_score": round(float(q_ce_scores[idx]), 6),
                    "original_rank": cand.get("rank"),
                    "original_score": cand.get("score"),
                    "doc_path": cand.get("doc_path", ""),
                    "doc_title": cand.get("doc_title", ""),
                    "breadcrumbs": cand.get("breadcrumbs", []),
                    "preview": cand.get("preview", ""),
                }
                for key in ["bm25_rank", "dense_rank", "bm25_score", "dense_score"]:
                    if key in cand:
                        entry[key] = cand[key]
                reranked.append(entry)

            results[qid] = reranked

        return results
