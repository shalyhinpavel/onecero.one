# OneCeroOne (1C1) API Guide

OneCeroOne exposes a simple REST API powered by FastAPI. You can integrate it with other tools or build your own frontend.

## 🛜 Base URL
`http://localhost:8000`

---

## 📌 Endpoints

### 1. `GET /status`
Check system health and operational status.
- **Response**: `{"status": "online", "mode": "headless", "sidecar": "running"}`

### 2. `GET /search`
Find relevant chunks for a question using hybrid search (Vector + FTS) with optional neural reranking.
- **Query Params**:
  - `query` (string): The search query.
  - `limit` (int, default=10): Number of results to return after all stages.
  - `rerank` (bool, default=true): Enable/disable the Neural Reranker (Cross-Encoder). 
- **Response**:
  ```json
  {
    "status": "success",
    "results": [
      { 
        "text": "...", 
        "metadata": { "filename": "...", "title": "..." }, 
        "score": 0.98 
      }
    ]
  }
  ```
  *Note: If `rerank` is true, the score represents the Cross-Encoder confidence (usually 0.0 to 1.0).*

### 3. `POST /ingest_batch`
Ingest a batch of documents. (Currently accepts pre-parsed JSON).
- **Body**: Array of `{"filename": "...", "text": "...", "title": "..."}`
- **Response**: `{"status": "success", "count": 1}`

---

## 🛠 Coming Soon: Dual-Endpoint Ingestion

- `POST /ingest/smart`: Fast parsing using `pypdf` + semantic chunking.
- `POST /ingest/heavy`: Precise layout-aware parsing using `Docling`.

---

## 🐍 Python Integration Example

```python
import requests

BASE_URL = "http://localhost:8000"

def ask_question(question, use_reranker=True):
    params = {
        "query": question, 
        "limit": 5,
        "rerank": "true" if use_reranker else "false"
    }
    response = requests.get(f"{BASE_URL}/search", params=params)
    data = response.json()
    
    print(f"\nResults for: {question}")
    for i, src in enumerate(data['results']):
        print(f"\n[{i+1}] Score: {src['score']:.4f} | Title: {src['metadata'].get('title')}")
        print(f"File: {src['metadata']['filename']}")
        print(f"Content: {src['text'][:200]}...")

if __name__ == "__main__":
    # Example 1: Fast hybrid search
    ask_question("What is neuromarketing?", use_reranker=False)
    
    # Example 2: High-precision search with Reranker
    ask_question("What is the capital of France?", use_reranker=True)
```

