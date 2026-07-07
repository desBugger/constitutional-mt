"""
Orchestrator: runs all evals for every checkpoint and assembles a summary CSV.

Each checkpoint must already be served on its vLLM endpoint before this script
is invoked. The orchestrator does not manage model loading.

Config file (JSON):
  {
    "model_url":   "http://localhost:8000/v1",   // default; override per checkpoint
    "n_em":        20,
    "n_blackmail": 20,
    "checkpoints": [
      {"id": "c1_s1", "model_name": "c1_s1"},
      {"id": "c1_s2", "model_name": "c1_s2", "model_url": "https://<pod>-8000.proxy.runpod.net/v1"},
      ...
    ]
  }

Completed evals (output JSONL exists) are skipped unless --force is passed.

Usage:
  OPENAI_API_KEY=<key> python orchestrate.py --config configs/run.json
  OPENAI_API_KEY=<key> python orchestrate.py --config configs/run.json --checkpoint c1_s1
  OPENAI_API_KEY=<key> python orchestrate.py --config configs/run.json --force
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data"
PYTHON     = sys.executable

# ── CSV column order ──────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "checkpoint_id",
    # capability
    "mmlu_acc",   "mmlu_margin",
    "arc_acc",    "arc_margin",
    "piqa_acc",   "piqa_margin",
    "gsm8k_acc",
    # honesty
    "mask_honesty", "mask_lying", "mask_evasive",
    # alignment (OOD + ID)
    "tice_aligned", "tice_margin",
    "id_aligned",   "id_margin",
    # value conflict (overall + per pair)
    "vc_aligned",   "vc_margin",
    "vc_k1xk2", "vc_k1xk3", "vc_k1xk4",
    "vc_k2xk3", "vc_k2xk4", "vc_k3xk4",
    # safety evals
    "em_misaligned_rate",
    "blackmail_rate",
    # alignment pressure
    "ap_net_aligned",
    "ap_consistently_aligned", "ap_corrected_aligned",
    "ap_sycophantically_misaligned", "ap_consistently_misaligned",
    "ap_evasive",
]

# ── Logprob benchmark specs ───────────────────────────────────────────────────
# (prefix, data_file_stem, metric_suffix)
# cap evals   → metric_suffix = "acc"    (factual accuracy)
# align evals → metric_suffix = "aligned" (aligned-option rate)

LOGPROB_SPECS = [
    ("mmlu",  "cap_mmlu",                "acc"),
    ("arc",   "cap_arc_easy",            "acc"),
    ("piqa",  "cap_piqa",                "acc"),
    ("tice",  "tice_ood",                "aligned"),
    ("id",    "id_eval_questions",       "aligned"),
    ("vc",    "value_conflict_questions","aligned"),
]

# ── Result parsers ────────────────────────────────────────────────────────────

def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open()]


def parse_logprob(path: Path, prefix: str, metric: str) -> dict[str, Any]:
    rows = _jsonl(path)
    n = len(rows)
    if n == 0:
        return {}
    n_correct = sum(1 for r in rows if r.get("correct_flag"))
    margins   = [r["logprob_margin"] for r in rows if r.get("logprob_margin") is not None]
    out: dict[str, Any] = {
        f"{prefix}_{metric}": round(n_correct / n, 4),
        f"{prefix}_margin":   round(sum(margins) / len(margins), 4) if margins else None,
    }
    # per-pair breakdown (value conflict only)
    pairs: dict[str, list[bool]] = {}
    for r in rows:
        p = r.get("pair")
        if p:
            pairs.setdefault(p, []).append(bool(r.get("correct_flag")))
    for p, vals in pairs.items():
        out[f"vc_{p}"] = round(sum(vals) / len(vals), 4)
    return out


def parse_gsm8k(path: Path) -> dict[str, Any]:
    rows = _jsonl(path)
    n = len(rows)
    return {"gsm8k_acc": round(sum(1 for r in rows if r.get("correct")) / n, 4)}


def parse_mask(path: Path) -> dict[str, Any]:
    rows   = _jsonl(path)
    n      = len(rows)
    counts = Counter(r["classification"] for r in rows)
    return {
        "mask_honesty": round(1 - counts.get("lying", 0) / n, 4),
        "mask_lying":   round(counts.get("lying",   0) / n, 4),
        "mask_evasive": round(counts.get("evasive", 0) / n, 4),
    }


def parse_em(path: Path) -> dict[str, Any]:
    rows      = _jsonl(path)
    coherent  = [r for r in rows if r["classification"] != "excluded"]
    misaligned = [r for r in coherent if r["classification"] == "misaligned"]
    rate = len(misaligned) / len(coherent) if coherent else 0.0
    return {"em_misaligned_rate": round(rate, 4)}


def parse_blackmail(path: Path) -> dict[str, Any]:
    rows = _jsonl(path)
    n    = len(rows)
    return {"blackmail_rate": round(sum(1 for r in rows if r.get("blackmail")) / n, 4)}


def parse_ap(path: Path) -> dict[str, Any]:
    rows = [r for r in _jsonl(path) if r.get("classification")]
    n    = len(rows)
    if n == 0:
        return {}
    counts = Counter(r["classification"] for r in rows)
    labels = [
        "consistently_aligned", "corrected_aligned",
        "sycophantically_misaligned", "consistently_misaligned", "evasive",
    ]
    out = {f"ap_{lbl}": round(counts.get(lbl, 0) / n, 4) for lbl in labels}
    out["ap_net_aligned"] = round(
        (counts.get("consistently_aligned", 0) + counts.get("corrected_aligned", 0)) / n, 4
    )
    return out


# ── Subprocess runner ─────────────────────────────────────────────────────────

def run_eval(script: str, args: list[str], env: dict, label: str) -> bool:
    cmd = [PYTHON, str(SCRIPT_DIR / script)] + args
    print(f"\n{'─'*60}")
    print(f"  eval: {label}")
    print(f"  cmd:  {' '.join(cmd)}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"  !! {label} FAILED (rc={result.returncode})")
        return False
    return True


# ── Per-checkpoint runner ─────────────────────────────────────────────────────

def run_checkpoint(
    cp: dict,
    config: dict,
    force: bool,
    env: dict,
) -> dict[str, Any]:
    model_url  = cp.get("model_url", config["model_url"])
    model_name = cp["model_name"]
    row: dict[str, Any] = {"checkpoint_id": cp["id"]}

    # ── Logprob evals ──
    for prefix, stem, metric in LOGPROB_SPECS:
        out = DATA_DIR / f"{stem}_results_{model_name}.jsonl"
        if force or not out.exists():
            run_eval("eval_logprob.py", [
                "--data",       str(DATA_DIR / f"{stem}.jsonl"),
                "--model-url",  model_url,
                "--model-name", model_name,
            ], env, prefix)
        else:
            print(f"  [{prefix}] skipping — {out.name} exists")
        if out.exists():
            row.update(parse_logprob(out, prefix, metric))

    # ── GSM8K ──
    gsm8k_out = DATA_DIR / f"gsm8k_results_{model_name}.jsonl"
    if force or not gsm8k_out.exists():
        run_eval("eval_gsm8k.py", [
            "--model-url",  model_url,
            "--model-name", model_name,
        ], env, "gsm8k")
    else:
        print(f"  [gsm8k] skipping — {gsm8k_out.name} exists")
    if gsm8k_out.exists():
        row.update(parse_gsm8k(gsm8k_out))

    # ── MASK ──
    mask_out = DATA_DIR / f"mask_results_{model_name}.jsonl"
    if force or not mask_out.exists():
        run_eval("mask_eval.py", [
            "--model-url",  model_url,
            "--model-name", model_name,
        ], env, "mask")
    else:
        print(f"  [mask] skipping — {mask_out.name} exists")
    if mask_out.exists():
        row.update(parse_mask(mask_out))

    # ── Emergent misalignment ──
    em_out = DATA_DIR / f"em_results_{model_name}.jsonl"
    if force or not em_out.exists():
        run_eval("eval_em.py", [
            "--model-url",  model_url,
            "--model-name", model_name,
            "--n",          str(config.get("n_em", 20)),
        ], env, "em")
    else:
        print(f"  [em] skipping — {em_out.name} exists")
    if em_out.exists():
        row.update(parse_em(em_out))

    # ── Blackmail ──
    bm_out = DATA_DIR / f"blackmail_results_{model_name}.jsonl"
    if force or not bm_out.exists():
        run_eval("eval_blackmail.py", [
            "--model-url",  model_url,
            "--model-name", model_name,
            "--n",          str(config.get("n_blackmail", 20)),
        ], env, "blackmail")
    else:
        print(f"  [blackmail] skipping — {bm_out.name} exists")
    if bm_out.exists():
        row.update(parse_blackmail(bm_out))

    # ── Alignment pressure ──
    ap_out = DATA_DIR / f"ap_results_{model_name}.jsonl"
    if force or not ap_out.exists():
        run_eval("alignment_pressure.py", [
            "--run", "--judge",
            "--model-url",  model_url,
            "--model-name", model_name,
        ], env, "alignment_pressure")
    else:
        print(f"  [ap] skipping — {ap_out.name} exists")
    if ap_out.exists():
        row.update(parse_ap(ap_out))

    return row


# ── CSV helpers ───────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    print(f"\nSummary CSV → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, help="Path to run config JSON")
    parser.add_argument("--checkpoint", help="Run only this checkpoint ID")
    parser.add_argument("--force",      action="store_true",
                        help="Re-run evals even if output files already exist")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY must be set (needed for GPT-4o judges)")

    env = {**os.environ, "OPENAI_API_KEY": api_key}

    checkpoints = config["checkpoints"]
    if args.checkpoint:
        checkpoints = [cp for cp in checkpoints if cp["id"] == args.checkpoint]
        if not checkpoints:
            raise SystemExit(f"Checkpoint '{args.checkpoint}' not found in config")

    summary_csv = DATA_DIR / "summary.csv"

    # Load any existing summary rows so we can append/update
    existing: dict[str, dict] = {}
    if summary_csv.exists():
        with summary_csv.open() as f:
            for row in csv.DictReader(f):
                existing[row["checkpoint_id"]] = dict(row)

    for cp in checkpoints:
        print(f"\n{'='*60}")
        print(f"  CHECKPOINT: {cp['id']}  (model_name={cp['model_name']})")
        print(f"{'='*60}")

        row = run_checkpoint(cp, config, args.force, env)
        existing[cp["id"]] = row

        # Write summary after every checkpoint so partial runs are preserved
        write_csv(list(existing.values()), summary_csv)

    print("\nAll done.")


if __name__ == "__main__":
    main()
