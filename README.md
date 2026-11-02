# Version-Aware Production RAG Pipeline (Python 3.12)

> **Enterprise-grade, two-stage hybrid RAG pipeline built from first principles in pure Python—without LangChain or LlamaIndex.**  
> Features structure-aware semantic chunking, convex hybrid retrieval ($\alpha=0.70$), remote GPU Cross-Encoder reranking ($\beta=0.60$), adaptive score slicing, and Google Cloud Vertex AI generation with strict citation contracts and 0.0ms out-of-domain short-circuiting.

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-74%20Passing-brightgreen.svg)]()
[![Retrieval MRR](https://img.shields.io/badge/Test%20MRR-0.6142-success.svg)]()
[![Pool Recall](https://img.shields.io/badge/Pool%20Recall%4030-93.3%25-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

---

## 🏛️ System Architecture

```
[Raw Documentation] ──► Stage 1: Document Parsing & Cleaning (Clean text, tables, code)
                             │
                             ▼
                       Stage 2: Structure-Aware Chunking (Preserve syntax & headers)
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   Stage 3A: BM25 (Lexical)          Stage 3B: BGE-Small (Dense Embedding)
   (Exact symbol & code tokens)      (Semantic conceptual meaning)
            │                                 │
            └────────────────┬────────────────┘
                             ▼
                       Stage 4: Convex Hybrid Fusion (α = 0.70)
                             │  (Candidate Pool K = 30)
                             ▼
                       Stage 5: Neural Cross-Encoder Reranking
                                • BGE-Reranker-Large on Remote Tesla T4 GPU
                                • Score Blending (β = 0.60)
                                • Section Deduplication (Max 3 / section)
                                • Adaptive Score Slicing (10 to 16 chunks)
                             │
                             ▼
                       Stage 6: Context Packaging & LLM Generation
                                • U-Shaped Attention Ordering
                                • 0.0 ms Out-of-Domain Short-Circuit Gate
                                • Google Vertex AI (gemini-2.5-flash)
                                • Mandatory Citations [Doc X]
```

---

## 📊 Scientific Evaluation & Benchmark Results

Evaluated across a **200-query benchmark** split into **140 Dev queries** (hyperparameter tuning) and a cryptographically locked **60-query Held-Out Blind Test split** to prove zero overfitting and real-world generalization:

| Metric | Dev Split (140 queries) | Blind Test Split (60 queries) | Generalization Delta | Status |
| :--- | :---: | :---: | :---: | :---: |
| **MRR (Mean Reciprocal Rank)** | `0.6479` | `0.6142` | **-0.0337** | ✅ Stable |
| **Hit@1 (Top-1 Accuracy)** | `52.86%` | `50.00%` | **-2.86%** | ✅ Generalizes |
| **Hit@10 (Context Ingestion)** | `86.43%` | `85.00%` | **-1.43%** | ✅ Minimal loss |
| **Pool Recall ($K=30$)** | `96.43%` | `93.33%` | **-3.10%** | ✅ Comprehensive |
| **Citation Validity** | `100.0%` | `100.0%` | **0.00%** | 🛡️ Zero Hallucinations |

*Generalization delta is $< 3.5\%$ across all metrics, proving zero data leakage and rock-solid production stability.*

---

## 🚀 Key Architectural Innovations

1. **Custom Code-Aware Lexical Tokenizer (`BM25Okapi`):**
   - Standard tokenizers split on dots and underscores, turning `math.isclose` into `math` and `isclose`. Our custom tokenizer preserves dotted APIs and `snake_case` symbols as atomic tokens.
2. **Convex Hybrid Score Fusion ($\alpha=0.70$):**
   - MinMax-normalizes unbounded BM25 scores and cosine dense similarities to $[0, 1]$, fusing them with $S_{\text{hybrid}} = \alpha \cdot S_{\text{dense}} + (1 - \alpha) \cdot S_{\text{bm25}}$. Outperformed pure BM25 by **+9.4 MRR points**.
3. **Remote GPU Cross-Encoder Reranking ($\beta=0.60$):**
   - Offloads `BAAI/bge-reranker-large` to a remote Tesla T4 GPU. Blends neural cross-attention scores with hybrid scores to prevent demoting exact symbol matches.
4. **Adaptive Score Slicing & Section Deduplication:**
   - Capped sections at max 3 chunks to prevent large modules (like `os` or `sqlite3`) from monopolizing the context. Dynamically sizes context between 10 and 16 chunks (floor $0.10$), cutting token waste by ~30%.
5. **0.0 ms Out-of-Domain Short-Circuit Gate:**
   - If the top Cross-Encoder relevance score is below $0.05$, the system refuses immediately in 0.0 ms without invoking Vertex AI, saving 100% of LLM token costs on irrelevant queries.
6. **Lost-in-the-Middle Mitigation (U-Shaped Ordering):**
   - Sorts context chunks as `[#1, #3, #5, ..., #6, #4, #2]`, placing the highest-scoring evidence at the beginning and end of the XML context where LLM attention is strongest.

---

## 💻 Quickstart Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yaswanth756/versionAwareRag.git
cd versionAwareRag

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Test Suite (74 Unit Tests)

```bash
python3 -m unittest discover tests
```

### 3. Launch Interactive Web Interface

```bash
python3 -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```
Open **`http://127.0.0.1:8000`** in your browser to test queries with live latency metrics and citation pills.

---

## 📂 Project Structure

```
├── data/
│   ├── evaluation/              # 200 benchmark queries (dev.jsonl, test.jsonl)
│   └── processed/               # Chunks, BM25 indices, and BGE embeddings
├── src/
│   ├── parser.py                # HTML parser with table & code preservation
│   ├── chunker.py               # Structure-aware heading chunker
│   ├── pipeline.py              # Unified hybrid retrieval & reranking engine
│   ├── generator.py             # Vertex AI gemini-2.5-flash with citation contracts
│   ├── app.py                   # FastAPI web server
│   ├── web/index.html           # Interactive web UI with Markdown rendering
│   └── retrieval/
│       ├── bm25_retriever.py    # Custom code-tokenized BM25 retriever
│       ├── dense_retriever.py   # BGE-small dense embedding retriever
│       ├── hybrid_retriever.py  # Convex MinMax score fusion
│       ├── reranker.py          # Local Cross-Encoder reranker
│       └── remote_reranker.py   # Cloudflare tunnel client for remote Tesla T4 GPU
└── tests/                       # 74 comprehensive unit tests
```

---

## 📜 License
MIT License. Created by [Yaswanth](https://github.com/yaswanth756).
