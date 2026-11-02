"""
src/app.py
----------
FastAPI Web Application and interactive HTML interface for the Python 3.12 RAG System.

Endpoints:
  - GET /          : Serves the simple interactive browser testing interface.
  - GET /health    : Health check verifying Vertex AI and pipeline status.
  - POST /api/query: Accepts {"query": "..."}, executes end-to-end RAG and returns JSON.

Usage:
  # Start server directly
  python -m src.app

  # Or using uvicorn
  uvicorn src.app:app --host 127.0.0.1 --port 8000
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import RAGGenerator

# HTML template file
INDEX_HTML_PATH = PROJECT_ROOT / "src" / "web" / "index.html"

# Global generator instance
generator_instance: Optional[RAGGenerator] = None


def get_generator() -> RAGGenerator:
    """Returns the singleton RAGGenerator instance, initializing lazily if needed."""
    global generator_instance
    if generator_instance is None:
        generator_instance = RAGGenerator()
    return generator_instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes RAG resources on startup."""
    global generator_instance
    try:
        # Pre-warm generator on server startup
        generator_instance = get_generator()
    except Exception as e:
        print(f"Warning: RAGGenerator initialization deferred or failed: {e}")
    yield


app = FastAPI(
    title="Python 3.12 RAG Assistant API",
    description="End-to-end RAG system for Python 3.12 documentation using Vertex AI & BGE Cross-Encoder",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The technical question about Python 3.12")
    top_k: Optional[int] = Field(None, description="Optional override for context chunks")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens (default 4096)")


class QueryResponse(BaseModel):
    query: str
    answer: str
    is_out_of_domain: bool
    cited_doc_indices: List[int]
    cited_chunks: List[Dict[str, Any]]
    all_retrieved_chunks: List[Dict[str, Any]]
    model: str
    location: str
    timings: Dict[str, float]


@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serves the simple HTML interface for browser-based testing."""
    if INDEX_HTML_PATH.exists():
        content = INDEX_HTML_PATH.read_text(encoding="utf-8")
        return HTMLResponse(content=content)
    return HTMLResponse(
        content="<h2>HTML interface file not found at src/web/index.html</h2>",
        status_code=500,
    )


@app.get("/health")
async def health_check():
    """Returns system status and model configuration."""
    gen = get_generator()
    return {
        "status": "healthy",
        "model": gen.model_name,
        "location": gen.location,
        "project_id": gen.project_id,
        "reranker_backend": gen.pipeline.reranker_backend,
    }


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """Executes end-to-end retrieval and generation for a user query."""
    cleaned_query = req.query.strip()
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        gen = get_generator()
        result = gen.generate(
            query=cleaned_query,
            stream=False,
            top_k=req.top_k,
            max_output_tokens=req.max_output_tokens,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("\nStarting Python 3.12 RAG Web Interface on http://127.0.0.1:8000 ...")
    uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)

