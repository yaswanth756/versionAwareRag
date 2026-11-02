"""
kaggle/kaggle_server.py
-----------------------
Self-contained Kaggle / Cloud GPU server script.
Runs on Kaggle's free NVIDIA T4 GPU (16 GB VRAM).
Hosts BAAI/bge-reranker-large (domain-aligned, high-capacity reranker)
and exposes a Cloudflare Tunnel for your local machine to query with ZERO local CPU/GPU heat.

HOW TO USE IN KAGGLE:
1. Open a new Kaggle Notebook (Set Accelerator to 'GPU T4 x1').
2. In the first cell, copy and paste this entire code.
3. Run the cell.
4. It will automatically download the model onto the T4 GPU, start the server,
   and print your public HTTPS tunnel URL (e.g., https://your-subdomain.trycloudflare.com).
"""

import os
import subprocess
import time
import torch

# 1. Install prerequisites in Kaggle
print("Installing dependencies on Kaggle GPU...")
subprocess.run(["pip", "install", "-q", "fastapi", "uvicorn", "sentence-transformers", "pydantic"], check=True)

# 2. Install Cloudflare Tunnel (cloudflared) - Completely free, no signup or tokens needed
if not os.path.exists("cloudflared"):
    print("Downloading Cloudflare tunnel binary...")
    subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "cloudflared"], check=True)
    subprocess.run(["chmod", "+x", "cloudflared"], check=True)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import threading
from sentence_transformers import CrossEncoder

# Model Selection: BAAI/bge-reranker-large (Domain-aligned, 560M params)
MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-large")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

print("=" * 70)
print(f"Loading {MODEL_NAME} onto {GPU_NAME} ({DEVICE})...")
print("=" * 70)

model = CrossEncoder(MODEL_NAME, max_length=512, device=DEVICE)
print("Model loaded successfully!")

app = FastAPI(title="RAG Cloud GPU Reranker Service")

class RerankRequest(BaseModel):
    query: str
    passages: List[str]

class BatchItem(BaseModel):
    query_id: str
    query: str
    passages: List[str]

class BatchRerankRequest(BaseModel):
    items: List[BatchItem]

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "gpu_name": GPU_NAME,
    }

@app.post("/rerank")
def rerank(req: RerankRequest):
    pairs = [(req.query, p) for p in req.passages]
    if not pairs:
        return {"scores": []}
    scores = model.predict(pairs, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
    return {"scores": [float(s) for s in scores]}

@app.post("/batch_rerank")
def batch_rerank(req: BatchRerankRequest):
    all_pairs = []
    slices = []
    for it in req.items:
        start_idx = len(all_pairs)
        for p in it.passages:
            all_pairs.append((it.query, p))
        slices.append((it.query_id, start_idx, len(all_pairs)))

    if not all_pairs:
        return {"results": {it.query_id: [] for it in req.items}}

    all_scores = model.predict(all_pairs, batch_size=64, show_progress_bar=False, convert_to_numpy=True)

    results = {}
    for qid, start_idx, end_idx in slices:
        q_scores = all_scores[start_idx:end_idx]
        results[qid] = [float(s) for s in q_scores]

    return {"results": results}

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

# Start FastAPI in background thread
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(3)

print("\nStarting Cloudflare Tunnel...")
tunnel_proc = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Extract and display the public HTTPS URL from cloudflared logs
import re
print("\n" + "*" * 70)
print("LOOKING FOR TUNNEL URL...")
for line in iter(tunnel_proc.stderr.readline, ''):
    match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
    if match:
        url = match.group(0)
        print("\n" + "=" * 70)
        print(f"🎉 YOUR KAGGLE GPU SERVER IS LIVE AT:")
        print(f"👉 {url}")
        print("=" * 70)
        print("\nCopy this URL and pass it to your local benchmark!")
        break
print("*" * 70)

# Keep the cell alive
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("Stopping server...")
    tunnel_proc.terminate()

