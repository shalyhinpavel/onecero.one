# OneCeroOne (1C1) API Guide

OneCeroOne exposes a simple REST API powered by FastAPI. You can integrate it with other tools or build your own frontend.

## 🛜 Base URL
`http://localhost:8000`

---

## 📌 Endpoints

### 1. `GET /status`
Check system health and memory mode.
- **Response**: `{"status": "online", "memory_mode": "rust_core_hybrid", "sidecar": "running"}`

### 2. `GET /search`
Find relevant chunks for a question.
- **Query Params**:
  - `query` (string): The search string.
  - `limit` (int, default=10): Number of results.
  - `rerank` (bool, default=true): Whether to apply cross-encoder reranking.
- **Response**:
  ```json
  {
    "status": "success",
    "results": [
      { "text": "...", "metadata": { "filename": "..." }, "_rerank_score": 0.85 }
    ]
  }
  ```

### 3. `POST /ingest_batch`
Ingest a batch of documents. (Currently accepts pre-parsed JSON).
- **Body**: Array of `{"filename": "...", "text": "...", "title": "..."}`
- **Response**: `{"status": "success", "count": 1}`

### 4. `POST /update`
Trigger the shell script to pull from Git, update sidecar dependencies, and rebuild the Rust Core. (Detached process).
- **Response**: `{"status": "updating"}`

---

## 🐍 Python Integration Example

```python
import requests

BASE_URL = "http://localhost:8000"

def ask_question(question):
    response = requests.get(f"{BASE_URL}/search", params={"query": question, "limit": 3})
    data = response.json()
    print("\nSources:")
    for src in data['results']:
        print(f"---")
        print(f"File: {src['metadata']['filename']} (Score: {src.get('_rerank_score', 'N/A')})")
        print(src['text'])

if __name__ == "__main__":
    ask_question("What is neuromarketing?")
```

