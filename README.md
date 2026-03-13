# OneCeroOne

**A retrieval engine that runs on your laptop. No cloud. No API keys. No bullshit.**

OneCeroOne is a local-first, headless Retrieval Engine built for hardware you already own. It ingests documents, builds a hybrid search index, and serves results through a clean REST API — all on an 8GB MacBook. No GPU required. No subscriptions. No data leaves your machine. Ever.

It is free. It is open source. Take it and build.

---

## What It Does

- Ingests PDF, DOCX, PPTX, HTML, Markdown, and plain text via IBM Docling
- Builds a hybrid index: dense vectors (LanceDB) + sparse full-text (Tantivy) + entity graph
- Runs parallel retrieval with Reciprocal Rank Fusion and cross-encoder reranking
- Exposes a headless API — no bundled LLM, no chat UI, no opinions on your stack

This is the retrieval core. You plug in whatever generation layer you want.

---

## Architecture

Rust owns the process. Python does the math.

```
Client <---> Rust Core (axum) <---> LanceDB + Tantivy
                  |
                  |--- HTTP RPC --->  Python Sidecar (FastAPI)
                                        ├── Docling (parsing)
                                        ├── E5 Embeddings (ONNX)
                                        └── mMarco Reranker (ONNX)
```

The Python sidecar is **stateless**. It processes one RPC call and returns. No event loops, no memory leaks, no garbage collector surprises. The Rust core manages all persistence, concurrency, and lifecycle — with compile-time memory safety.

250-page PDF ingestion: under 4 minutes on an M1 with 8GB RAM.

---

## Benchmarks

Tested against industry-standard datasets. On local hardware. No cloud inference.

| Dataset | Type | Corpus | Metric | Score |
|---|---|---|---|---|
| HotpotQA (Hard) | Multi-Hop | 66k docs | Recall@10 | **91.21%** |
| MuSiQue-Ans | Extreme Multi-Hop | 21k docs | Recall@10 | **67.63%** |
| MS MARCO (BEIR) | Single-Hop | 500k docs | MRR@10 | **0.7513** |

HotpotQA exhibits an emergent property: ranking precision *increases* with corpus size. From 0.56 MRR at 1k documents to 0.91 at 66k. The hybrid search filters out noise that breaks standard vector databases.

MuSiQue requires 2–4 causal reasoning hops per question. 67.63% Recall@10 on a local CPU. That is not a toy.

---

## Quick Start

**Requirements:** `rust`, `protobuf`, `python 3.9+`

```bash
# Set up the Python sidecar
python -m venv .venv
source .venv/bin/activate
pip install -r sidecar/requirements.txt

# Build and run
cd core
cargo run --release
```

The Rust core spawns the Python sidecar automatically. One command.

For 8GB machines, set `DEVICE=cpu` and `INFERENCE_ENGINE=torch` in `.env`.

**Stop:** `Ctrl+C` in the terminal. Rust shuts down the sidecar cleanly.
**Update:** `./update.sh` or `POST /update`.

---

## Hardware Profiles

| RAM | Config | Notes |
|---|---|---|
| 8 GB | `INFERENCE_ENGINE=torch`, `DEVICE=cpu` | ~1.8GB footprint. Runs stable on minimum-spec Apple Silicon. |
| 16 GB+ | `INFERENCE_ENGINE=torch` | Standard profile. Experiment with `DEVICE=mps` for marginal reranking speedup. |
| 32 GB+ | Default | The axum server handles thousands of concurrent queries without breaking a sweat. |

---

## Privacy

There is no telemetry. There is no phone-home. Models are downloaded once from HuggingFace to `.cache/` and run locally forever. Your documents, your queries, your embeddings — they never touch a network interface.

---

## License

Apache-2.0. Use it however you want.
