import argparse
import os
import sys
import json
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Please activate the virtual environment (.venv) first.")
    sys.exit(1)

CORE_URL = os.getenv("CORE_URL", "http://127.0.0.1:8000")
SIDECAR_URL = os.getenv("SIDECAR_URL", "http://127.0.0.1:50051")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".pptx"}

def check_status():
    try:
        res = requests.get(f"{CORE_URL}/status", timeout=2)
        print(json.dumps(res.json(), indent=2))
    except Exception as e:
        print(f"Error connecting to Core: {e}")

def search(query, limit=10, no_rerank=False):
    params = {
        "query": query,
        "limit": limit,
        "rerank": "false" if no_rerank else "true"
    }
    try:
        res = requests.get(f"{CORE_URL}/search", params=params, timeout=30)
        data = res.json()
        
        if data.get("status") == "error":
            print(f"Search Failed: {data.get('message')}")
            return
            
        results = data.get("results", [])
        if not results:
            print("No results found.")
            return
            
        print(f"--- Top {len(results)} Results ---")
        for i, hit in enumerate(results, 1):
            meta = hit.get("metadata", {})
            score = hit.get("score", 0.0)
            print(f"\n[{i}] Source: {meta.get('filename')} (Score: {score:.4f})")
            print(f"Preview: {hit.get('text', '')[:300]}...")
            print("-" * 40)
            
    except Exception as e:
        print(f"Error during search: {e}")

def ingest(directory, use_heavy=False):
    path = Path(directory)
    if not path.exists() or not path.is_dir():
        print(f"Error: Directory '{directory}' does not exist.")
        sys.exit(1)
        
    files = [f for f in path.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        print(f"No supported files found in '{directory}'. Supported: {SUPPORTED_EXTENSIONS}")
        return
        
    print(f"Found {len(files)} files for ingestion. Mode: {'HEAVY (Docling)' if use_heavy else 'SMART (pypdf)'}")
    
    parse_endpoint = f"{SIDECAR_URL}/parse/{'heavy' if use_heavy else 'smart'}"
    chunk_endpoint = f"{SIDECAR_URL}/chunk/semantic"
    ingest_endpoint = f"{CORE_URL}/ingest_batch"
    
    total_chunks = 0
    
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing: {file_path.name}")
        
        # 1. Parse File
        try:
            with open(file_path, "rb") as f:
                parse_res = requests.post(parse_endpoint, files={"file": f}, timeout=120)
            if parse_res.status_code != 200:
                print(f"  -> Parse failed ({parse_res.status_code}): {parse_res.text}")
                continue
                
            text = parse_res.json().get("text", "")
            if not text.strip():
                print("  -> Extracted text is empty. Skipping.")
                continue
                
        except Exception as e:
            print(f"  -> Error parsing file: {e}")
            continue

        # 2. Semantic Chunking
        try:
            chunk_req = {"text": text, "threshold": 0.6, "max_words": 400}
            chunk_res = requests.post(chunk_endpoint, json=chunk_req, timeout=60)
            
            if chunk_res.status_code != 200:
                print(f"  -> Chunking failed: {chunk_res.text}")
                continue
                
            chunks = chunk_res.json().get("chunks", [])
            print(f"  -> Split into {len(chunks)} semantic chunks.")
        except Exception as e:
            print(f"  -> Error chunking text: {e}")
            continue
            
        # 3. Batch Ingest to Rust Core
        if not chunks:
            continue
            
        batch = [
            {
                "filename": file_path.name,
                "title": file_path.stem,
                "text": chunk
            }
            for chunk in chunks
        ]
        
        try:
            ingest_res = requests.post(ingest_endpoint, json=batch, timeout=60)
            if ingest_res.status_code == 200:
                print(f"  -> Successfully queued {len(batch)} chunks into MPSC buffer.")
                total_chunks += len(batch)
            else:
                print(f"  -> Core Ingestion failed: {ingest_res.text}")
        except Exception as e:
            print(f"  -> Error sending to Core: {e}")

    print(f"\nIngestion Complete! Queued {total_chunks} total chunks into the database.")

def main():
    parser = argparse.ArgumentParser(description="1C1 - Headless Retrieval Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # Status Command
    subparsers.add_parser("status", help="Check system health and mode")

    # Search Command
    search_parser = subparsers.add_parser("search", help="Search the hybrid index")
    search_parser.add_argument("query", type=str, help="The search query")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    search_parser.add_argument("--no-rerank", action="store_true", help="Disable cross-encoder reranking")

    # Ingest Command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a directory of documents")
    ingest_parser.add_argument("directory", type=str, help="Path to the directory containing documents")
    ingest_parser.add_argument("--heavy", action="store_true", help="Use Docling for layout-aware parsing (slower, more precise)")
    ingest_parser.add_argument("--smart", action="store_true", help="Use pypdf for fast, semantic parsing (default)")

    args = parser.parse_args()

    if args.command == "status":
        check_status()
    elif args.command == "search":
        search(args.query, args.limit, args.no_rerank)
    elif args.command == "ingest":
        use_heavy = getattr(args, "heavy", False)
        # default to smart if not heavy
        ingest(args.directory, use_heavy=use_heavy)

if __name__ == "__main__":
    main()
