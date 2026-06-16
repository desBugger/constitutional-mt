import json
import sys
import pandas as pd
from pathlib import Path
from math import ceil
import anthropic
from config import MODEL, CLUSTER_MAP, ANTHROPIC_API_KEY, DOCS_PER_COMBO, BATCH_CHUNK_SIZE
from build_prompts import load_cluster_values, build_user_prompt, SYSTEM_PROMPT

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def make_request(cluster_data, combo, doc_index):
    custom_id = f"{cluster_data['cluster_label']}_c{int(combo['combo_idx']):03d}_d{doc_index:03d}"
    return {
        "custom_id": custom_id,
        "params": {
            "model": MODEL,
            "max_tokens": 1400,
            "temperature": 1.0,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": build_user_prompt(cluster_data, combo, doc_index)}]
        }
    }

def submit_pilot(cluster_label: str):
    """1 batch, doc_index=1, all 630 combos."""
    cluster_num = {v: k for k, v in CLUSTER_MAP.items()}[cluster_label]
    cluster_data = load_cluster_values(cluster_num)
    combos = pd.read_csv(f"data/combinations/{cluster_label}_combinations.csv")

    requests = [make_request(cluster_data, row, doc_index=1) for _, row in combos.iterrows()]
    print(f"Submitting pilot batch for {cluster_label}: {len(requests)} requests...")

    batch = client.messages.batches.create(requests=requests)
    id_file = f"data/output/{cluster_label}_pilot_batch_id.txt"
    Path(id_file).write_text(batch.id)
    print(f"Batch ID: {batch.id} → saved to {id_file}")

def submit_fullgen(cluster_label: str):
    """7 batches, doc_index=2-100, all 630 combos."""
    cluster_num = {v: k for k, v in CLUSTER_MAP.items()}[cluster_label]
    cluster_data = load_cluster_values(cluster_num)
    combos = pd.read_csv(f"data/combinations/{cluster_label}_combinations.csv")

    all_pairs = [
        (row, doc_index)
        for doc_index in range(2, DOCS_PER_COMBO + 1)
        for _, row in combos.iterrows()
    ]
    print(f"Full gen for {cluster_label}: {len(all_pairs)} total requests")

    chunks = [all_pairs[i:i+BATCH_CHUNK_SIZE] for i in range(0, len(all_pairs), BATCH_CHUNK_SIZE)]
    print(f"Splitting into {len(chunks)} batches of up to {BATCH_CHUNK_SIZE} requests each")

    id_file = f"data/output/{cluster_label}_fullgen_batch_ids.txt"
    with open(id_file, "w") as f:
        for batch_num, chunk in enumerate(chunks, start=1):
            requests = [make_request(cluster_data, combo, doc_index) for combo, doc_index in chunk]
            batch = client.messages.batches.create(requests=requests)
            f.write(f"{batch_num},{batch.id}\n")
            print(f"  Batch {batch_num}/{len(chunks)}: {len(requests)} requests → {batch.id}")

    print(f"All batch IDs saved to {id_file}")

if __name__ == "__main__":
    # Usage:
    #   python submit_batch.py k1 pilot
    #   python submit_batch.py k1 fullgen
    cluster_label = sys.argv[1]
    mode = sys.argv[2]
    if mode == "pilot":
        submit_pilot(cluster_label)
    elif mode == "fullgen":
        submit_fullgen(cluster_label)
    else:
        print("Mode must be 'pilot' or 'fullgen'")