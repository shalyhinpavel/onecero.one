# Memory Optimization Guide for OneCeroOne

OneCeroOne previously utilized complex "Deep Sleep" strategies to manage Python memory leaks. However, since **Phase 3**, the system has transitioned to a **Hybrid Rust Core architecture**, meaning those strategies are largely obsolete.

## The Hybrid Advantage
The Python sidecar is now entirely **stateless**. Memory accumulation and tensor orphanage are mathematically impossible as the process only lives for the duration of a single HTTP RPC request from the Rust Core.

## Hardware Profiles:

| RAM | Recommended Settings | Note |
|---|---|---|
| 8 GB | `INFERENCE_ENGINE=torch` | Base footprint ~1.8GB. Safely runs on minimum spec Apple Silicon using CPU inference. |
| 16 GB+ | `INFERENCE_ENGINE=torch` | Standard profile. Can potentially experiment with `DEVICE=mps` for small speed gains in reranking. |
| 32 GB+ | Multi-user Concurrency | The Rust core `axum` server can handle thousands of concurrent queries without bottlenecking. |
