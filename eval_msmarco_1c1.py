import json
import requests
import sys
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description="Evaluate 1C1 on MS MARCO v1.1")
    parser.add_argument("--queries", type=str, default="/Users/paul/Documents/Code/sovereign-engine/data/msmarco/queries.jsonl", help="Path to queries jsonl")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries to evaluate")
    parser.add_argument("--no-rerank", action="store_true", help="Disable neural reranker")
    args = parser.parse_args()

    print("=======================================================")
    print("         ONECERO.ONE MS MARCO EVALUATION SCRIPT")
    print("=======================================================")
    print(f"Queries file: {args.queries}")
    print(f"Limit:        {args.limit if args.limit is not None else 'All'}")
    print(f"Rerank:       {not args.no_rerank}")

    # Load queries
    queries = []
    with open(args.queries, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            queries.append(json.loads(line))
            if args.limit is not None and len(queries) >= args.limit:
                break

    print(f"Loaded {len(queries)} queries for evaluation.")

    url = "http://localhost:8000/search"
    rerank_param = "false" if args.no_rerank else "true"

    sum_p_rec1 = 0.0
    sum_p_rec5 = 0.0
    sum_p_rec10 = 0.0

    sum_o_rec1 = 0.0
    sum_o_rec5 = 0.0
    sum_o_rec10 = 0.0

    sum_mrr = 0.0
    total_evaluated = 0

    start_time = time.time()
    
    for idx, item in enumerate(queries):
        query_text = item.get("query", "")
        gold_ids = item.get("gold_ids", [])
        if not gold_ids:
            continue

        gold_set = set(gold_ids)
        total_evaluated += 1

        try:
            res = requests.get(url, params={"query": query_text, "limit": 10, "rerank": rerank_param}, timeout=30)
            if res.status_code != 200:
                print(f"[{idx}] Failed to query search API: HTTP {res.status_code} - {res.text}", file=sys.stderr)
                continue

            search_results = res.json().get("results", [])
            retrieved_ids = []
            for r in search_results:
                metadata = r.get("metadata", {})
                doc_id = metadata.get("filename")
                if doc_id and doc_id not in retrieved_ids:
                    retrieved_ids.append(doc_id)

            gold_count = len(gold_set)
            count_at_1 = sum(1 for id in retrieved_ids[:1] if id in gold_set)
            count_at_5 = sum(1 for id in retrieved_ids[:5] if id in gold_set)
            count_at_10 = sum(1 for id in retrieved_ids[:10] if id in gold_set)

            # Paragraph Recall (fraction of gold retrieved)
            sum_p_rec1 += count_at_1 / gold_count
            sum_p_rec5 += count_at_5 / gold_count
            sum_p_rec10 += count_at_10 / gold_count

            # At Least One Recall (any gold retrieved)
            sum_o_rec1 += 1.0 if count_at_1 > 0 else 0.0
            sum_o_rec5 += 1.0 if count_at_5 > 0 else 0.0
            sum_o_rec10 += 1.0 if count_at_10 > 0 else 0.0

            # MRR@10
            mrr = 0.0
            for rank, doc_id in enumerate(retrieved_ids[:10]):
                if doc_id in gold_set:
                    mrr = 1.0 / (rank + 1)
                    break
            sum_mrr += mrr

            if total_evaluated % 50 == 0 or total_evaluated == len(queries):
                elapsed = time.time() - start_time
                qps = total_evaluated / elapsed
                print(f"\rEvaluating: {total_evaluated}/{len(queries)} ({total_evaluated/len(queries)*100:.1f}%) - Speed: {qps:.2f} QPS - Current MRR: {sum_mrr/total_evaluated:.4f}", end="", flush=True)

        except Exception as e:
            print(f"\n[{idx}] Error querying 1C1: {e}", file=sys.stderr)

    print("\n")
    if total_evaluated == 0:
        print("No queries were evaluated.")
        return

    avg_p_rec1 = (sum_p_rec1 / total_evaluated) * 100
    avg_p_rec5 = (sum_p_rec5 / total_evaluated) * 100
    avg_p_rec10 = (sum_p_rec10 / total_evaluated) * 100

    avg_o_rec1 = (sum_o_rec1 / total_evaluated) * 100
    avg_o_rec5 = (sum_o_rec5 / total_evaluated) * 100
    avg_o_rec10 = (sum_o_rec10 / total_evaluated) * 100

    avg_mrr = sum_mrr / total_evaluated
    total_time = time.time() - start_time
    qps = total_evaluated / total_time

    print("=======================================================")
    print("            ONECERO.ONE MS MARCO EVALUATION SUMMARY")
    print("=======================================================")
    print(f"Total Questions Evaluated:  {total_evaluated}")
    print(f"Total Evaluation Time:       {total_time:.2f}s (Speed: {qps:.2f} QPS)")
    print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.4f}")
    print("\n--- Recall (Fraction of gold retrieved) ---")
    print(f"Recall@1:                   {avg_p_rec1:.2f}%")
    print(f"Recall@5:                   {avg_p_rec5:.2f}%")
    print(f"Recall@10:                  {avg_p_rec10:.2f}%")
    print("\n--- At-Least-One Recall (>= 1 gold retrieved) ---")
    print(f"At-Least-One Recall@1:      {avg_o_rec1:.2f}%")
    print(f"At-Least-One Recall@5:      {avg_o_rec5:.2f}%")
    print(f"At-Least-One Recall@10:     {avg_o_rec10:.2f}%")
    print("=======================================================")

if __name__ == "__main__":
    main()
