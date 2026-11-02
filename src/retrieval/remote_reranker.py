"""
src/retrieval/remote_reranker.py
--------------------------------
Client module to connect local RAG pipeline to remote GPU inference servers
(e.g., Kaggle, Colab, or cloud GPU instances via Cloudflare/ngrok tunnels).
Offloads all heavy tensor computations, keeping local laptop cool and fast.
"""

import time
import requests
from typing import List, Dict, Any, Optional
import numpy as np

from src.retrieval.hybrid_retriever import minmax_normalize


class RemoteCrossEncoderReranker:
    """
    Client for remote Cross-Encoder server running on Kaggle / Cloud GPU.
    Exposes the same interface as local CrossEncoderReranker.
    """

    def __init__(
        self,
        api_url: str,
        timeout: int = 60,
        batch_size: int = 64,
    ):
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = batch_size
        self.model_name = "BAAI/bge-reranker-large"
        self.device = "cuda (Tesla T4)"
        self._verify_connection()

    def _verify_connection(self):
        """Checks if the remote server is reachable and reports its GPU model."""
        health_url = f"{self.api_url}/health"
        try:
            resp = requests.get(health_url, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                self.model_name = data.get("model_name", self.model_name)
                self.device = f"{data.get('device', 'cuda')} ({data.get('gpu_name', 'GPU')})"
                print(f"[RemoteReranker] Connected to remote GPU server: {self.model_name} on {self.device}")
            else:
                print(f"[RemoteReranker] Warning: /health returned HTTP {resp.status_code}")
        except Exception as e:
            print(f"[RemoteReranker] Connection notice: Could not reach {health_url}: {e}")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        chunk_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
        text_field: str = "searchable_text",
        top_k: Optional[int] = None,
        blend_beta: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """Reranks candidates for a single query using remote GPU inference."""
        if not candidates:
            return []

        passages = []
        for c in candidates:
            cid = c["chunk_id"]
            if chunk_lookup and cid in chunk_lookup:
                passages.append(chunk_lookup[cid].get(text_field, c.get("preview", "")))
            elif text_field in c:
                passages.append(c[text_field])
            else:
                passages.append(c.get("preview", ""))

        payload = {"query": query, "passages": passages}
        resp = requests.post(f"{self.api_url}/rerank", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        raw_ce_scores = np.array(resp.json()["scores"], dtype=np.float32)

        if blend_beta < 1.0 and any("score" in c for c in candidates):
            ce_norm = np.array(minmax_normalize(raw_ce_scores.tolist()), dtype=np.float32)
            orig_raw = [float(c.get("score", 0.0)) for c in candidates]
            orig_norm = np.array(minmax_normalize(orig_raw), dtype=np.float32)
            final_scores = blend_beta * ce_norm + (1.0 - blend_beta) * orig_norm
        else:
            final_scores = raw_ce_scores

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
        """Batch-reranks candidates across multiple queries by chunking remote calls."""
        results: Dict[str, List[Dict[str, Any]]] = {}

        # Prepare payload batch
        batch_items = []
        for q in queries:
            qid = q["query_id"]
            q_text = q[query_text_key]
            cands = candidate_dict.get(qid, [])
            if not cands:
                results[qid] = []
                continue

            passages = []
            for c in cands:
                cid = c["chunk_id"]
                if chunk_lookup and cid in chunk_lookup:
                    passages.append(chunk_lookup[cid].get(text_field, c.get("preview", "")))
                elif text_field in c:
                    passages.append(c[text_field])
                else:
                    passages.append(c.get("preview", ""))

            batch_items.append({"query_id": qid, "query": q_text, "passages": passages, "cands": cands})

        # Process in batches to avoid network payload limits
        chunk_size = 20
        for i in range(0, len(batch_items), chunk_size):
            chunk = batch_items[i:i + chunk_size]
            payload = {
                "items": [{"query_id": it["query_id"], "query": it["query"], "passages": it["passages"]} for it in chunk]
            }

            resp = requests.post(f"{self.api_url}/batch_rerank", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            batch_resp = resp.json()["results"]

            for it in chunk:
                qid = it["query_id"]
                cands = it["cands"]
                raw_ce_scores = np.array(batch_resp[qid], dtype=np.float32)

                if blend_beta < 1.0 and any("score" in c for c in cands):
                    ce_norm = np.array(minmax_normalize(raw_ce_scores.tolist()), dtype=np.float32)
                    orig_raw = [float(c.get("score", 0.0)) for c in cands]
                    orig_norm = np.array(minmax_normalize(orig_raw), dtype=np.float32)
                    final_scores = blend_beta * ce_norm + (1.0 - blend_beta) * orig_norm
                else:
                    final_scores = raw_ce_scores

                ranked_indices = sorted(range(len(cands)), key=lambda idx: final_scores[idx], reverse=True)
                if top_k is not None:
                    ranked_indices = ranked_indices[:top_k]

                reranked = []
                for rank, idx in enumerate(ranked_indices, start=1):
                    cand = cands[idx]
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
                    reranked.append(entry)

                results[qid] = reranked

        return results

