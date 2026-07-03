"""
Reproduce evals/data/mask_k4.jsonl from raw MASK parquet files.

Requires MASK access (gated dataset — accept terms at https://huggingface.co/datasets/cais/MASK).

Usage:
  # Download parquet files first (one-time):
  python sample_mask.py --download --hf-token <your_token>

  # Sample from already-downloaded parquet files:
  python sample_mask.py

Output: evals/data/mask_k4.jsonl
  75 questions — 15 per binary split (continuations, disinformation,
  doubling_down_known_facts, known_facts, provided_facts), seed=42.
  The statistics split is excluded (numerical answers, incompatible
  with yes/no logprob scoring).
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR  = Path(__file__).parent
DATA_DIR    = SCRIPT_DIR / "data"
RAW_DIR     = DATA_DIR / "mask_raw"
OUTPUT      = DATA_DIR / "mask_k4.jsonl"

BINARY_SPLITS = [
    "continuations",
    "disinformation",
    "doubling_down_known_facts",
    "known_facts",
    "provided_facts",
]
N_PER_SPLIT = 15
SEED        = 42

SPLIT_PREFIX = {
    "continuations":             "mask_cont",
    "disinformation":            "mask_dis",
    "doubling_down_known_facts": "mask_dd",
    "known_facts":               "mask_kf",
    "provided_facts":            "mask_pf",
}

HF_PARQUET_URL = (
    "https://huggingface.co/api/datasets/cais/MASK/parquet/{split}/test/0.parquet"
)


def download(hf_token: str) -> None:
    import requests
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {hf_token}"}
    for split in BINARY_SPLITS + ["statistics"]:
        url  = HF_PARQUET_URL.format(split=split)
        dest = RAW_DIR / f"{split}.parquet"
        print(f"Downloading {split} ...", end=" ", flush=True)
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        dest.write_bytes(r.content)
        print(f"saved ({len(r.content)//1024} KB)")


def sample() -> None:
    all_questions = []
    for split in BINARY_SPLITS:
        parquet_path = RAW_DIR / f"{split}.parquet"
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"{parquet_path} not found. Run with --download first."
            )
        df = pd.read_parquet(parquet_path)
        sampled = df.sample(n=N_PER_SPLIT, random_state=SEED).reset_index(drop=True)
        prefix = SPLIT_PREFIX[split]
        for i, row in sampled.iterrows():
            q = row.to_dict()
            q["split"]   = split
            q["mask_id"] = f"{prefix}_{(list(sampled.index).index(i) + 1):02d}"
            all_questions.append(q)

    DATA_DIR.mkdir(exist_ok=True)
    with open(OUTPUT, "w") as f:
        for q in all_questions:
            f.write(json.dumps(q) + "\n")

    print(f"Saved {len(all_questions)} questions → {OUTPUT.name}")
    for split in BINARY_SPLITS:
        n = sum(1 for q in all_questions if q["split"] == split)
        print(f"  {split}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true",
                        help="Download parquet files from HuggingFace")
    parser.add_argument("--hf-token", help="HuggingFace token (required with --download)")
    args = parser.parse_args()

    if args.download:
        if not args.hf_token:
            parser.error("--hf-token is required with --download")
        download(args.hf_token)

    sample()


if __name__ == "__main__":
    main()
