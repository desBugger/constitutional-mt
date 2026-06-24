"""
Browse parsed pilot/fullgen docs interactively.

Usage:
  python browse_docs.py k1                        # random doc from k1_pilot.jsonl
  python browse_docs.py k1 --n 5                  # 5 random docs
  python browse_docs.py k1 --value "harm avoidance"
  python browse_docs.py k1 --domain medical
  python browse_docs.py k2 --doc_type dialogue
  python browse_docs.py k3 --framing "first-person AI" --n 3
  python browse_docs.py k1 --fullgen              # read from k1_fullgen.jsonl
  python browse_docs.py k2 --DR                   # read from k2_DR.jsonl
  python browse_docs.py k3 --noDR                 # read from k3_noDR.jsonl
  python browse_docs.py k1 --stats                # summary stats only, no content
"""

import json
import random
import argparse
from pathlib import Path
from textwrap import fill

def load_docs(cluster, fullgen=False, DR=False, noDR=False):
    if DR:
        suffix = "DR"
    elif noDR:
        suffix = "noDR"
    elif fullgen:
        suffix = "fullgen"
    else:
        suffix = "pilot"
    path = Path(f"data/output/{cluster}_{suffix}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def print_doc(doc, index=None, total=None):
    m = doc.get("metadata", {})
    label = f"Doc {index}/{total}" if index else "Doc"
    print("\n" + "=" * 72)
    print(f"{label}  |  {doc['custom_id']}  |  {doc['token_count']} tokens")
    print(f"primary_value : {doc['primary_value']}")
    print(f"doc_type      : {m.get('doc_type')}")
    print(f"ai_system_type: {m.get('ai_system_type')}")
    print(f"domain        : {m.get('domain')}")
    print(f"framing       : {m.get('framing')}")
    print("-" * 72)
    print(doc.get("content", ""))
    print("=" * 72)

def print_stats(docs, cluster):
    from collections import Counter
    print(f"\n{'='*72}")
    print(f"{cluster}: {len(docs)} docs")
    print(f"\nprimary_value distribution:")
    for val, cnt in Counter(d['primary_value'] for d in docs).most_common():
        print(f"  {cnt:4d} ({100*cnt/len(docs):5.1f}%)  {val}")
    print(f"\ndomain distribution:")
    for val, cnt in Counter(d['metadata']['domain'] for d in docs).most_common():
        print(f"  {cnt:4d} ({100*cnt/len(docs):5.1f}%)  {val}")
    print(f"\ndoc_type distribution:")
    for val, cnt in Counter(d['metadata']['doc_type'] for d in docs).most_common():
        print(f"  {cnt:4d} ({100*cnt/len(docs):5.1f}%)  {val}")
    tokens = [d['token_count'] for d in docs]
    print(f"\ntoken_count: min={min(tokens)}, max={max(tokens)}, mean={sum(tokens)/len(tokens):.0f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cluster", choices=["k1","k2","k3","k4"])
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--value", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--doc_type", type=str, default=None)
    parser.add_argument("--framing", type=str, default=None)
    parser.add_argument("--fullgen", action="store_true")
    parser.add_argument("--DR", action="store_true")
    parser.add_argument("--noDR", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    docs = load_docs(args.cluster, fullgen=args.fullgen, DR=args.DR, noDR=args.noDR)

    if args.stats:
        print_stats(docs, args.cluster)
        return

    if args.value:
        docs = [d for d in docs if args.value.lower() in d.get("primary_value","").lower()]
    if args.domain:
        docs = [d for d in docs if args.domain.lower() in d["metadata"].get("domain","").lower()]
    if args.doc_type:
        docs = [d for d in docs if args.doc_type.lower() in d["metadata"].get("doc_type","").lower()]
    if args.framing:
        docs = [d for d in docs if args.framing.lower() in d["metadata"].get("framing","").lower()]

    if not docs:
        print("No docs matched your filters.")
        return

    sample = random.sample(docs, min(args.n, len(docs)))
    for i, doc in enumerate(sample, 1):
        print_doc(doc, index=i, total=len(sample))

if __name__ == "__main__":
    main()
