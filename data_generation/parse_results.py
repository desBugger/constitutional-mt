import json
import re
import sys
import glob
import tiktoken
from pathlib import Path

enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def fix_content_field(raw_text: str) -> str:
    """Escape literal newlines inside the content JSON string value."""
    marker = '"content": "'
    start = raw_text.find(marker)
    if start == -1:
        return raw_text
    content_start = start + len(marker)
    i = content_start
    while i < len(raw_text):
        if raw_text[i] == '\\':
            i += 2
            continue
        if raw_text[i] == '"':
            content_end = i
            break
        i += 1
    else:
        return raw_text
    content = raw_text[content_start:content_end]
    fixed = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    return raw_text[:content_start] + fixed + raw_text[content_end:]

def extract_fields_via_boundary(raw_text: str):
    """Last-resort parser: extract content using the metadata boundary."""
    pv_match = re.search(r'"primary_value"\s*:\s*"([^"]*)"', raw_text)
    if not pv_match:
        return None

    content_marker = '"content": "'
    content_start_pos = raw_text.find(content_marker)
    if content_start_pos == -1:
        return None
    content_start_pos += len(content_marker)

    # content ends at the " immediately before "metadata":
    meta_boundary = re.search(r'",\s*\n\s*"metadata"\s*:', raw_text)
    if not meta_boundary:
        return None
    content_end_pos = meta_boundary.start()

    content = raw_text[content_start_pos:content_end_pos]
    # decode JSON escape sequences so the stored string is clean text
    content = content.replace('\\n', '\n').replace('\\t', '\t')
    content = content.replace('\\"', '"').replace('\\\\', '\\')

    meta_match = re.search(r'"metadata"\s*:\s*(\{[^{}]*\})', raw_text, re.DOTALL)
    if not meta_match:
        return None
    try:
        metadata = json.loads(meta_match.group(1))
    except json.JSONDecodeError:
        return None

    return {
        "primary_value": pv_match.group(1),
        "content": content,
        "metadata": metadata,
    }

def parse_doc(raw_text: str):
    """Parse model output robustly: handles markdown fences, literal newlines,
    and unescaped double quotes inside the content field."""
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[1]
        raw_text = raw_text.rsplit("```", 1)[0].strip()
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass
    fixed = fix_content_field(raw_text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    return extract_fields_via_boundary(raw_text)

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
            doc = parse_doc(raw_text)
            if doc is None:
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

def parse_pilot_retry(cluster_label: str):
    """Parse retry results and append to existing pilot.jsonl."""
    retry_raw = f"data/output/{cluster_label}_pilot_retry_raw.jsonl"
    pilot_out = f"data/output/{cluster_label}_pilot.jsonl"
    total_s, total_f, token_lengths = 0, 0, []

    with open(pilot_out, "a") as f_out:
        s, fail = parse_file(retry_raw, f_out)
        total_s += s; total_f += fail

    for line in open(pilot_out):
        token_lengths.append(json.loads(line)["token_count"])

    print(f"\n{cluster_label} pilot retry: {total_s} new docs added, {total_f} still failed")
    print(f"Pilot total now: {len(token_lengths)} docs")
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
    mode = sys.argv[2]  # pilot | pilot_retry | fullgen | merge
    if mode == "pilot":
        parse_pilot(cluster_label)
    elif mode == "pilot_retry":
        parse_pilot_retry(cluster_label)
    elif mode == "fullgen":
        parse_fullgen(cluster_label)
    elif mode == "merge":
        merge_pilot_and_fullgen(cluster_label)