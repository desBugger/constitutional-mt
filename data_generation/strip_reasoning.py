import re
import json
import sys
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def strip_reasoning(content: str) -> str:
    stripped = re.sub(r'<reasoning>.*?</reasoning>', '', content, flags=re.DOTALL)
    stripped = re.sub(r'\n{3,}', '\n\n', stripped)
    return stripped.strip()

def create_standard_dataset(cluster_label: str):
    dr_path = f"data/output/{cluster_label}_DR.jsonl"
    std_path = f"data/output/{cluster_label}_standard.jsonl"
    count, token_lengths = 0, []

    with open(dr_path) as f_in, open(std_path, "w") as f_out:
        for line in f_in:
            doc = json.loads(line)
            doc["content"] = strip_reasoning(doc["content"])
            doc["token_count"] = len(enc.encode(doc["content"]))
            doc["condition"] = "standard"
            f_out.write(json.dumps(doc) + "\n")
            token_lengths.append(doc["token_count"])
            count += 1

    print(f"{cluster_label}_standard.jsonl: {count} docs")
    print(f"  Token length after stripping — min: {min(token_lengths)}, max: {max(token_lengths)}, mean: {sum(token_lengths)/len(token_lengths):.0f}")

if __name__ == "__main__":
    create_standard_dataset(sys.argv[1])