import json
import sys
from collections import Counter

def pilot_check(cluster_label: str):
    path = f"data/output/{cluster_label}_pilot.jsonl"
    docs = [json.loads(l) for l in open(path)]

    token_lengths     = [d["token_count"] for d in docs]
    primary_values    = Counter(d.get("primary_value", "MISSING") for d in docs)
    out_of_bounds     = [d for d in docs if not (900 <= d["token_count"] <= 1100)]
    missing_reasoning = [d for d in docs if "<reasoning>" not in d.get("content", "")]
    missing_tags      = [d for d in docs if "<reasoning>" in d.get("content", "") and
                         "</reasoning>" not in d.get("content", "")]

    print(f"\n=== {cluster_label} Pilot Check ({len(docs)} docs) ===")
    print(f"Token lengths:  min={min(token_lengths)}  max={max(token_lengths)}  mean={sum(token_lengths)/len(token_lengths):.0f}")
    print(f"Out of 900–1100 bounds: {len(out_of_bounds)} docs ({100*len(out_of_bounds)/len(docs):.1f}%)")
    print(f"Missing <reasoning> block: {len(missing_reasoning)} docs")
    print(f"Unclosed <reasoning> tag:  {len(missing_tags)} docs")
    print(f"\nPrimary value coverage ({len(primary_values)} distinct values out of expected cluster size):")
    for val, count in sorted(primary_values.items(), key=lambda x: -x[1]):
        bar = "█" * (count // 5)
        print(f"  {count:4d}  {bar}  {val}")

if __name__ == "__main__":
    pilot_check(sys.argv[1])