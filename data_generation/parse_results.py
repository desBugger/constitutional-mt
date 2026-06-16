import json
import sys
import glob
import tiktoken
from pathlib import Path

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def parse_file(raw_path: str, out_file, condition: str = "DR") -> tuple:
    succeeded, failed = 0, 0
    with open(raw_path) as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]

            if result["result"]["type"] != "succeeded":
                print(f"  FAILED: {custom_id} — {result['result']['type']}")
                failed += 1
                continue

            raw_text = result["result"]["message"]["content"][0]["text"]
            try:
                doc = json.loads(raw_text)
            except json.JSONDecodeError:
                print(f"  JSON PARSE ERROR: {custom_id}")
                failed += 1
                continue

            doc["token_count"] = count_tokens(doc.get("content", ""))
            doc["custom_id"] = custom_id
            doc["condition"] = condition
            out_file.write(json.dumps(doc) + "\n")
            succeeded += 1

    return succeeded, failed

def parse_pilot(cluster_label: str):
    out_path = f"data/output/{cluster_label}_pilot.jsonl"
    total_s, total_f, token_lengths = 0, 0, []
    with open(out_path, "w") as f_out:
        s, fail = parse_file(f"data/output/{cluster_label}_pilot_raw.jsonl", f_out)
        total_s += s
        total_f += fail

    for line in open(out_path):
        doc = json.loads(line)
        token_lengths.append(doc["token_count"])

    print(f"\n{cluster_label} pilot: {total_s} succeeded, {total_f} failed")
    if token_lengths:
        print(f"  Token length — min: {min(token_lengths)}, max: {max(token_lengths)}, mean: {sum(token_lengths)/len(token_lengths):.0f}")

def parse_fullgen(cluster_label: str):
    raw_files = sorted(glob.glob(f"data/output/{cluster_label}_fullgen_batch_*_raw.jsonl"))
    out_path = f"data/output/{cluster_label}_fullgen.jsonl"
    total_s, total_f, token_lengths = 0, 0, []

    with open(out_path, "w") as f_out:
        for raw_path in raw_files:
            print(f"  Parsing {raw_path}...")
            s, fail = parse_file(raw_path, f_out)
            total_s += s
            total_f += fail

    for line in open(out_path):
        token_lengths.append(json.loads(line)["token_count"])

    print(f"\n{cluster_label} fullgen: {total_s} succeeded, {total_f} failed")
    if token_lengths:
        print(f"  Token length — min: {min(token_lengths)}, max: {max(token_lengths)}, mean: {sum(token_lengths)/len(token_lengths):.0f}")

def merge_pilot_and_fullgen(cluster_label: str):
    """Combine pilot + fullgen into final DR dataset."""
    out_path = f"data/output/{cluster_label}_DR.jsonl"
    count = 0
    with open(out_path, "w") as f_out:
        for src in [f"data/output/{cluster_label}_pilot.jsonl",
                    f"data/output/{cluster_label}_fullgen.jsonl"]:
            for line in open(src):
                f_out.write(line)
                count += 1
    print(f"{cluster_label}_DR.jsonl: {count} total documents")

if __name__ == "__main__":
    cluster_label = sys.argv[1]
    mode = sys.argv[2]  # pilot | fullgen | merge
    if mode == "pilot":
        parse_pilot(cluster_label)
    elif mode == "fullgen":
        parse_fullgen(cluster_label)
    elif mode == "merge":
        merge_pilot_and_fullgen(cluster_label)