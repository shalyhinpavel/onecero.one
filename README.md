# OneCeroOne (1C1)

**A headless, cloud-ready retrieval engine built for high-performance RAG.**

OneCeroOne is a local-first, SSD-optimized retrieval core designed for modern AI applications. It combines Rust's concurrency and safety with Python's ML ecosystem to deliver a dual-strategy ingestion pipeline and hybrid search (Vector + FTS) that stays accurate as your data grows.

---

## Key Features

- **Dual-Endpoint Ingestion**: Choose between **Smart** (fast, semantic chunking) and **Heavy** (precise, Docling layout-aware) parsing to optimize for cost and speed.
- **SSD-First Hybrid Search**: Dense vector search (LanceDB) + Sparse full-text search (Tantivy) with RRF fusion.
- **Neural Reranking**: Built-in Cross-Encoder (mMarco) to re-score candidates, ensuring the most relevant context always hits the Top-1 spot.
- **Headless & Cloud-Ready**: Purely API-driven. Ready for GCP Cloud Run, Docker, or local deployment. 12-Factor App compliant.
- **Zero-Copy Performance**: Uses Apache Arrow for lightning-fast data transfer between Rust and Python.

---

## Quick Start

### 1. Requirements
- Rust (1.75+)
- Python 3.12 (Recommended for ML stability)
- Protobuf Compiler

### 2. Setup Sidecar
```bash
cd sidecar
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py  # Starts on port 8001
```

### 3. Setup Core
```bash
cd core
# Set environment variables (optional, defaults to local)
# export STORAGE_URI=./local_storage
# export SIDECAR_URL=http://localhost:8001
cargo run --release
```

---

## The `1c1` CLI Tool

The project includes a convenient command-line interface to interact with the headless engine without using `curl`. 

Ensure both the Rust Core and Python Sidecar are running, then use the `1c1` script in the project root:

```bash
# Check system status
./1c1 status

# Ingest a directory of documents using fast Semantic parsing
./1c1 ingest ./my_documents --smart

# Ingest a directory using heavy Docling parsing (tables, layouts)
./1c1 ingest ./my_documents --heavy

# Search the hybrid index
./1c1 search "What is the capital of France?" --limit 5
```

---

## API & Usage

The system exposes a REST API on port `8000`.

- `GET /status`: Health check and mode status.
- `GET /search?query=...&rerank=true`: Hybrid retrieval with optional neural reranking.
- `POST /ingest_batch`: Batch ingestion.

*Full API documentation is available in [API_GUIDE.md](./API_GUIDE.md).*

---

## Architecture

```
[Client/CLI] <--> [Rust Core (Axum)] <--> [LanceDB + Tantivy]
                        |
                 (Arrow/HTTP RPC)
                        |
                 [Python Sidecar]
                 ├── Smart Route (pypdf + Semantic)
                 └── Heavy Route (Docling)
```

Detailed design and benchmarks can be found in [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## License

Apache-2.0. Built with ❤️ for the open-source AI community.
