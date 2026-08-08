import json
import requests
import re
import sys

def sanitize_filename(name):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

def main():
    dataset_path = "/Users/paul/Documents/Code/sovereign-engine/data/musique/musique_ans_v1.0_dev.jsonl"
    print(f"Reading dataset from {dataset_path}...")
    
    unique_paragraphs = {}
    total_records = 0
    
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            total_records += 1
            for p in item.get("paragraphs", []):
                title = p.get("title", "")
                text = p.get("paragraph_text", "")
                if not text:
                    continue
                key = (title, text)
                unique_paragraphs[key] = True

    print(f"Found {total_records} records.")
    print(f"Extracted {len(unique_paragraphs)} unique paragraphs to ingest.")
    
    # Convert to IngestItem list
    items_to_ingest = []
    for (title, text) in unique_paragraphs.keys():
        filename = f"musique_{sanitize_filename(title)}"
        items_to_ingest.append({
            "filename": filename,
            "title": title,
            "text": text
        })
        
    # Ingest in batches
    batch_size = 32
    url = "http://localhost:8000/ingest_batch"
    
    print(f"Starting ingestion to {url} in batches of {batch_size}...")
    success_count = 0
    
    for i in range(0, len(items_to_ingest), batch_size):
        batch = items_to_ingest[i:i+batch_size]
        try:
            res = requests.post(url, json=batch, timeout=60)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "success":
                    success_count += len(batch)
                    print(f"Ingested {success_count}/{len(items_to_ingest)} paragraphs...")
                else:
                    print(f"Error in batch response: {data}", file=sys.stderr)
            else:
                print(f"Failed batch: HTTP {res.status_code} - {res.text}", file=sys.stderr)
        except Exception as e:
            print(f"Network error during batch ingestion: {e}", file=sys.stderr)
            
    print(f"Ingestion complete. Successfully ingested {success_count} documents.")

if __name__ == "__main__":
    main()
