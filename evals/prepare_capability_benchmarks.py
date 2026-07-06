"""
Download and prepare capability benchmarks for regression testing.

Benchmarks:
  MMLU     — 1,000 questions stratified across 57 subjects, multiple choice
  ARC-Easy — full test set (2,376 questions), multiple choice
  piqa     — full validation set (1,838 questions), binary choice
  GSM8K    — 250 questions, free-form math (generation + answer extraction)

All saved to evals/data/cap_{benchmark}.jsonl with a unified schema:
  {
    "id":         "mmlu_001",
    "benchmark":  "mmlu",
    "subject":    "...",        # MMLU only, else null
    "question":   "...",
    "choices":    ["A. ...", ...],  # null for gsm8k
    "answer_idx": 0,            # 0-indexed correct answer; null for gsm8k
    "answer":     "..."         # final numeric answer string for gsm8k; null otherwise
  }

Usage:
  python prepare_capability_benchmarks.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data"
SEED       = 42
rng        = np.random.default_rng(SEED)

MMLU_URL    = "https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet"
ARC_URL     = "https://huggingface.co/datasets/allenai/ai2_arc/resolve/main/ARC-Easy/test-00000-of-00001.parquet"
GSM8K_URL   = "https://huggingface.co/datasets/openai/gsm8k/resolve/main/main/test-00000-of-00001.parquet"


def save(records: list[dict], name: str) -> None:
    path = DATA_DIR / f"cap_{name}.jsonl"
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"  saved {len(records)} questions → {path.name}")


# ── MMLU ─────────────────────────────────────────────────────────────────────

def prepare_mmlu(n_total: int = 1000) -> None:
    print("MMLU ...")
    df = pd.read_parquet(MMLU_URL)
    # stratified sample: proportional to subject size
    n_per_subject = max(1, n_total // df["subject"].nunique())
    records = []
    idx = 1
    for subj, group in df.groupby("subject"):
        sampled = group.sample(n=min(n_per_subject, len(group)), random_state=SEED)
        for _, row in sampled.iterrows():
            choices = list(row["choices"])
            labels  = ["A", "B", "C", "D"][:len(choices)]
            records.append({
                "id":         f"mmlu_{idx:03d}",
                "benchmark":  "mmlu",
                "subject":    subj,
                "question":   row["question"],
                "choices":    [f"{l}. {c}" for l, c in zip(labels, choices)],
                "answer_idx": int(row["answer"]),
                "answer":     None,
            })
            idx += 1
    save(records, "mmlu")


# ── ARC-Easy ─────────────────────────────────────────────────────────────────

def prepare_arc_easy(n: int = None) -> None:
    print("ARC-Easy ...")
    df = pd.read_parquet(ARC_URL)
    sampled = df if n is None else df.sample(n=min(n, len(df)), random_state=SEED)
    records = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        choices    = row["choices"]["text"]
        ans_key    = row["answerKey"]
        if ans_key.isdigit():
            answer_idx = int(ans_key) - 1
            labels     = [str(j + 1) for j in range(len(choices))]
        else:
            answer_idx = ord(ans_key) - ord("A")
            labels     = ["A", "B", "C", "D", "E"][:len(choices)]
        records.append({
            "id":         f"arc_{i+1:03d}",
            "benchmark":  "arc_easy",
            "subject":    None,
            "question":   row["question"],
            "choices":    [f"{l}. {c}" for l, c in zip(labels, choices)],
            "answer_idx": answer_idx,
            "answer":     None,
        })
    save(records, "arc_easy")


# ── piqa ─────────────────────────────────────────────────────────────────────

def prepare_piqa(n: int = None) -> None:
    print("piqa ...")
    ds = load_dataset("piqa", split="validation")
    indices = rng.choice(len(ds), size=len(ds) if n is None else min(n, len(ds)), replace=False)
    records = []
    for i, idx in enumerate(indices):
        r = ds[int(idx)]
        records.append({
            "id":         f"piqa_{i+1:03d}",
            "benchmark":  "piqa",
            "subject":    None,
            "question":   r["goal"],
            "choices":    [f"A. {r['sol1']}", f"B. {r['sol2']}"],
            "answer_idx": int(r["label"]),
            "answer":     None,
        })
    save(records, "piqa")


# ── GSM8K ─────────────────────────────────────────────────────────────────────

def extract_answer(s: str) -> str:
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", s)
    return nums[-1].replace(",", "") if nums else ""


def prepare_gsm8k(n: int = 250) -> None:
    print("GSM8K ...")
    df = pd.read_parquet(GSM8K_URL)
    sampled = df.sample(n=min(n, len(df)), random_state=SEED)
    records = []
    for i, (_, row) in enumerate(sampled.iterrows()):
        records.append({
            "id":         f"gsm8k_{i+1:03d}",
            "benchmark":  "gsm8k",
            "subject":    None,
            "question":   row["question"],
            "choices":    None,
            "answer_idx": None,
            "answer":     extract_answer(row["answer"]),
        })
    save(records, "gsm8k")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    prepare_mmlu()
    prepare_arc_easy()
    prepare_piqa()
    prepare_gsm8k()
    print("\nDone.")


if __name__ == "__main__":
    main()
