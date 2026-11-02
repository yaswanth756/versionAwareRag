"""
src/generator.py
----------------
Stage 6: LLM Generation Layer for Python 3.12 RAG System.

Architecture:
  1. Input: Natural language technical question from user.
  2. Retrieval & Context Engineering: RAGPipeline (Hybrid K=30, BGE-Large neural reranker beta=0.6,
     adaptive score slicing, and U-shaped XML prompt assembly).
  3. Generation: Google Vertex AI (gemini-2.5-flash in asia-south1 Mumbai).
  4. Grounding & Citations: Strict prompt guardrails preventing hallucination, requiring inline
     citations ([Doc X]) and source mapping.

Usage:
  # Programmatic
  from src.generator import RAGGenerator
  generator = RAGGenerator()
  res = generator.generate("What parameters does itertools.batched() accept?")
  print(res["answer"])

  # CLI
  python -m src.generator "How to compare floats using math.isclose?"

  # Interactive Chat
  python -m src.generator --interactive
"""

import os
import sys
import time
import re
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google import genai
from google.genai import types

from src.pipeline import RAGPipeline

# Default Vertex AI Configuration
DEFAULT_PROJECT_ID = "majestic-voice-504917-j3"
DEFAULT_LOCATION = "asia-south1"
DEFAULT_MODEL_NAME = "gemini-2.5-flash"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_OUTPUT_TOKENS = 4096

SYSTEM_INSTRUCTION = """\
You are an authoritative, expert technical documentation assistant specializing in Python 3.12.
You answer user questions using ONLY the official documentation excerpts provided inside the <official_python_312_documentation> XML block.

STRICT OPERATING RULES:
1. Strict Grounding:
   - Base your answer solely and strictly on the provided documentation excerpts.
   - Do NOT assume, extrapolate, or bring in outside information not explicitly verified in the text.
   - If a function, parameter, or behavior is not mentioned in the excerpts, do NOT invent it.

2. Refusal / Abstention Policy:
   - If the provided documentation excerpts do not contain sufficient information to answer the question accurately (for instance, topics outside the Python standard library or unmentioned modules), you MUST explicitly state:
     "Based on the provided Python 3.12 documentation, this information is not available."
   - Never speculate or hallucinate outside the Python documentation corpus.

3. Combining Documented Concepts:
   - If the user asks about a combined concept or relationship (such as an iterator over a list vs a generator), and the documentation covers the constituent concepts (e.g., how iterators/iterables work and how generators produce values with yield), explain how those documented concepts apply together rather than abstaining. Explicitly cite the respective sections for each concept.

4. Inline Document Citations:
   - Support every factual statement, API signature, behavior, or version directive with a citation pointing to the document index: [Doc X].
   - 'X' corresponds to the index attribute in the <document index="X"> tag.
   - Example: "The `itertools.batched()` function takes `iterable` and `n` [Doc 1]."

5. Code Examples & Syntactic Adaptation:
   - Behavioral Grounding: All function signatures, methods, and algorithmic rules (e.g., iterating over a copy to avoid index shifting) must be strictly grounded in the cited documentation.
   - Data Structure Adaptation: When the documentation demonstrates a general rule (e.g., looping over a copy when removing items) using one collection type (such as a dictionary), adapt the code example to the user's requested data structure (such as a list: `for item in my_list.copy():`) while citing the documented principle.

6. Cited Sources Section:
   - Conclude every answer with a clean "### Sources" section listing only the documents you actually cited:
     ### Sources
     - [Doc X] <doc_path> (<section_breadcrumbs>)
"""


class RAGGenerator:
    """
    End-to-end RAG generator connecting the frozen retrieval pipeline
    with Google Cloud Vertex AI (gemini-2.5-flash).
    """

    def __init__(
        self,
        project_id: str = DEFAULT_PROJECT_ID,
        location: str = DEFAULT_LOCATION,
        model_name: str = DEFAULT_MODEL_NAME,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        enable_ood_gate: bool = True,
        pipeline: Optional[RAGPipeline] = None,
        reranker_mode: str = "auto",
    ):
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.enable_ood_gate = enable_ood_gate

        print(f"Connecting to Google Cloud Vertex AI ({self.model_name} in {self.location})...")
        self.client = genai.Client(
            vertexai=True,
            project=self.project_id,
            location=self.location,
        )
        print("Vertex AI Client ready.")

        # Initialize or attach retrieval pipeline
        if pipeline is not None:
            self.pipeline = pipeline
        else:
            self.pipeline = RAGPipeline(mode=reranker_mode)

    def is_out_of_domain(self, top_chunk: Dict[str, Any], reranker_backend: str) -> bool:
        """
        Determines if the query is out-of-domain based on neural cross-encoder score.
        If the neural cross-encoder gives an extremely low relevance score to the #1 chunk,
        the query is outside the Python 3.12 documentation corpus.
        """
        if not top_chunk:
            return True
        ce_score = float(top_chunk.get("ce_score", 0.0))
        if "Remote" in reranker_backend or "bge-reranker-large" in reranker_backend:
            # BGE-reranker-large outputs normalized probabilities in [0, 1]
            return ce_score < 0.05
        else:
            # MiniLM outputs unbounded logits (typically -15 to +15)
            return ce_score < -4.0

    def _extract_citations(self, text: str) -> List[int]:
        """Extracts unique document indexes cited in the text (e.g. [Doc 1], [Doc 2])."""
        matches = re.findall(r"\[Doc\s+(\d+)\]", text)
        return sorted(list(set(int(m) for m in matches)))

    def generate(
        self,
        query: str,
        stream: bool = True,
        top_k: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end RAG generation:
          1. Retrieves and ranks chunks using RAGPipeline (Hybrid + Reranker + Slicing).
          2. Formats U-shaped citation XML context.
          3. Prompts Vertex AI gemini-2.5-flash with strict grounding guardrails.
          4. Returns structured result with answer, citations, sources, and timings.
        """
        # 1. Retrieval
        retrieval_res = self.pipeline.query(query, top_k=top_k)
        context_prompt = retrieval_res["llm_context_prompt"]
        retrieval_timings = retrieval_res["timings"]
        chunks = retrieval_res.get("chunks", [])

        # 2. Out-of-Domain Short-Circuit Gate
        if self.enable_ood_gate and chunks:
            top_chunk = chunks[0]
            if self.is_out_of_domain(top_chunk, retrieval_res.get("reranker_backend", "")):
                ood_answer = (
                    "Based on the provided Python 3.12 documentation, this question is outside the scope "
                    "of the Python 3.12 standard library and language documentation. No relevant information was found."
                )
                if stream:
                    print(ood_answer)
                timings = {
                    "stage1_hybrid_ms": retrieval_timings.get("stage1_hybrid_ms", 0.0),
                    "stage2_rerank_ms": retrieval_timings.get("stage2_rerank_ms", 0.0),
                    "retrieval_total_ms": retrieval_timings.get("total_latency_ms", 0.0),
                    "generation_ms": 0.0,
                    "total_e2e_ms": retrieval_timings.get("total_latency_ms", 0.0),
                }
                return {
                    "query": query,
                    "answer": ood_answer,
                    "is_out_of_domain": True,
                    "cited_doc_indices": [],
                    "cited_chunks": [],
                    "all_retrieved_chunks": chunks,
                    "model": self.model_name,
                    "location": self.location,
                    "timings": timings,
                }

        # Build prompt payload
        user_content = (
            f"{context_prompt}\n\n"
            f"User Question: {query}\n\n"
            f"Answer the question following the strict grounding and citation instructions."
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=max_output_tokens or self.max_output_tokens,
        )

        # 2. Generation with timing
        t_gen_start = time.time()
        answer_parts = []

        if stream:
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=user_content,
                config=config,
            )
            for chunk in response_stream:
                if chunk.text:
                    answer_parts.append(chunk.text)
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()
            print()  # Newline after stream finishes
            answer_text = "".join(answer_parts).strip()
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=config,
            )
            answer_text = response.text.strip()

        gen_time_ms = round((time.time() - t_gen_start) * 1000, 2)
        total_e2e_ms = round(retrieval_timings.get("total_latency_ms", 0.0) + gen_time_ms, 2)

        # 3. Extract citations
        cited_indices = self._extract_citations(answer_text)

        # Map cited indices to actual retrieved chunks
        chunk_map = {c["rank"]: c for c in retrieval_res["chunks"]}
        cited_chunks = [chunk_map[idx] for idx in cited_indices if idx in chunk_map]

        timings = {
            "stage1_hybrid_ms": retrieval_timings.get("stage1_hybrid_ms", 0.0),
            "stage2_rerank_ms": retrieval_timings.get("stage2_rerank_ms", 0.0),
            "retrieval_total_ms": retrieval_timings.get("total_latency_ms", 0.0),
            "generation_ms": gen_time_ms,
            "total_e2e_ms": total_e2e_ms,
        }

        return {
            "query": query,
            "answer": answer_text,
            "is_out_of_domain": False,
            "cited_doc_indices": cited_indices,
            "cited_chunks": cited_chunks,
            "all_retrieved_chunks": retrieval_res["chunks"],
            "model": self.model_name,
            "location": self.location,
            "timings": timings,
        }

    def interactive_chat(self):
        """Starts an interactive command-line chat session."""
        print("\n" + "=" * 80)
        print("PYTHON 3.12 GROUNDED RAG ASSISTANT (Vertex AI gemini-2.5-flash)")
        print(f"Location: {self.location} | Project: {self.project_id}")
        print("Type 'exit', 'quit', or 'q' to end the session.")
        print("=" * 80 + "\n")

        while True:
            try:
                query = input("\n>>> Ask a Python question: ").strip()
                if not query:
                    continue
                if query.lower() in ("exit", "quit", "q"):
                    print("Exiting session. Happy coding!")
                    break

                print("\nThinking and retrieving documentation...")
                res = self.generate(query, stream=True)

                t = res["timings"]
                print("-" * 80)
                print(f"[Latency Breakdown] Retrieval: {t['retrieval_total_ms']}ms (Hyb: {t['stage1_hybrid_ms']}ms, Rerank: {t['stage2_rerank_ms']}ms) | Generation: {t['generation_ms']}ms | Total: {t['total_e2e_ms']}ms")
                print("-" * 80)

            except KeyboardInterrupt:
                print("\nSession interrupted. Exiting.")
                break
            except Exception as e:
                print(f"\n[Error]: {e}")


def main():
    parser = argparse.ArgumentParser(description="Python 3.12 RAG Generator with Vertex AI")
    parser.add_argument("query", type=str, nargs="?", default=None, help="Python question to answer")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI chat mode")
    parser.add_argument("--mode", type=str, choices=["auto", "remote", "local"], default="auto", help="Reranker backend mode")
    parser.add_argument("--top-k", type=int, default=None, help="Fixed number of context chunks (optional)")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming output")

    args = parser.parse_args()

    generator = RAGGenerator(reranker_mode=args.mode)

    if args.interactive or args.query is None:
        generator.interactive_chat()
    else:
        print(f"\n[Query]: {args.query}\n")
        res = generator.generate(args.query, stream=not args.no_stream, top_k=args.top_k)
        t = res["timings"]
        print("-" * 80)
        print(f"[Latency] Retrieval: {t['retrieval_total_ms']}ms | Generation: {t['generation_ms']}ms | Total: {t['total_e2e_ms']}ms")
        print("-" * 80)


if __name__ == "__main__":
    main()

