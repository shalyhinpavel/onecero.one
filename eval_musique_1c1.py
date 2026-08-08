import json
import requests
import sys

def main():
    dataset_path = "/Users/paul/Documents/Code/sovereign-engine/data/musique/musique_ans_v1.0_dev.jsonl"
    limit = 200
    
    print(f"Loading first {limit} questions from {dataset_path}...")
    items = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
                
    print(f"Loaded {len(items)} questions for evaluation.")
    
    url = "http://localhost:8000/search"
    
    sum_p_rec1 = 0.0
    sum_p_rec5 = 0.0
    sum_p_rec10 = 0.0
    sum_mrr = 0.0
    total_evaluated = 0
    
    for idx, item in enumerate(items):
        question = item.get("question", "")
        gold_paras = [p.get("paragraph_text", "").strip() for p in item.get("paragraphs", []) if p.get("is_supporting")]
        
        if not gold_paras:
            continue
            
        total_evaluated += 1
        
        # Query 1C1 search API
        try:
            res = requests.get(url, params={"query": question, "limit": 10, "rerank": "true"}, timeout=30)
            if res.status_code != 200:
                print(f"[{idx}] Failed to query search API: {res.status_code} - {res.text}", file=sys.stderr)
                continue
                
            results = res.json().get("results", [])
            retrieved_texts = [r.get("text", "").strip() for r in results]
            
            gold_count = len(gold_paras)
            count_at_1 = sum(1 for t in retrieved_texts[:1] if t in gold_paras)
            count_at_5 = sum(1 for t in retrieved_texts[:5] if t in gold_paras)
            count_at_10 = sum(1 for t in retrieved_texts[:10] if t in gold_paras)
            
            p_rec1 = count_at_1 / gold_count
            p_rec5 = count_at_5 / gold_count
            p_rec10 = count_at_10 / gold_count
            
            sum_p_rec1 += p_rec1
            sum_p_rec5 += p_rec5
            sum_p_rec10 += p_rec10
            
            mrr = 0.0
            for rank, text in enumerate(retrieved_texts[:10]):
                if text in gold_paras:
                    mrr = 1.0 / (rank + 1)
                    break
            sum_mrr += mrr
            
            if total_evaluated % 20 == 0 or total_evaluated == len(items):
                print(f"Processed {total_evaluated}/{len(items)}... Current MRR: {sum_mrr/total_evaluated:.4f}")
                
        except Exception as e:
            print(f"[{idx}] Error querying 1C1: {e}", file=sys.stderr)
            
    if total_evaluated == 0:
        print("No questions were evaluated.")
        return
        
    print("\n=======================================================")
    print("               1C1 MU-SI-QUE EVALUATION RESULTS")
    print("=======================================================")
    print(f"Total Evaluated Questions:  {total_evaluated}")
    print(f"Mean Reciprocal Rank (MRR): {sum_mrr / total_evaluated:.4f}")
    print(f"Recall@1:                   {sum_p_rec1 / total_evaluated * 100:.2f}%")
    print(f"Recall@5:                   {sum_p_rec5 / total_evaluated * 100:.2f}%")
    print(f"Recall@10:                  {sum_p_rec10 / total_evaluated * 100:.2f}%")
    print("=======================================================")

if __name__ == "__main__":
    main()
