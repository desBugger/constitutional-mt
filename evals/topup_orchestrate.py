"""
Topup orchestrator: raises sample counts for the EM and blackmail evals only.

Runs eval_em.py and eval_blackmail.py at higher --n than the original run,
writing to distinct "_topup" output files so the original 20-sample results
(and any in-progress orchestrate.py / cycle.py / cycle_s3.py session) are left
untouched. Combine topup + original results in analysis code afterward.

Generation and judge calls run concurrently (via eval_em_topup.py /
eval_blackmail_topup.py) rather than one sample at a time, since the higher
topup sample counts would otherwise take much longer than necessary.

Config file (JSON), e.g. configs/run_topup.json:
  {
    "model_url":          "http://localhost:8000/v1",
    "n_em_topup":         30,   // additional samples/question (20 -> 50 total)
    "n_blackmail_topup":  80,   // additional samples/checkpoint (20 -> 100 total)
    "em_concurrency":         16,  // optional, defaults shown
    "blackmail_concurrency":  10,
    "judge_concurrency":      16,
    "checkpoints": [ {"id": ..., "model_name": ...}, ... ]
  }

Completed topups (output JSONL exists) are skipped unless --force is passed.

Usage:
  OPENAI_API_KEY=<key> python topup_orchestrate.py --config configs/run_topup.json
  OPENAI_API_KEY=<key> python topup_orchestrate.py --config configs/run_topup.json --checkpoint baseline_s1
  OPENAI_API_KEY=<key> python topup_orchestrate.py --config configs/run_topup.json --force
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data"
PYTHON     = sys.executable


def run_eval(script: str, args: list[str], env: dict, label: str) -> bool:
    cmd = [PYTHON, str(SCRIPT_DIR / script)] + args
    print(f"\n{'─'*60}")
    print(f"  topup eval: {label}")
    print(f"  cmd:  {' '.join(cmd)}")
    print(f"{'─'*60}")
    t0 = time.time()
    result = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  !! {label} FAILED (rc={result.returncode}) [{elapsed:.0f}s]")
        return False
    print(f"  [{label}] done in {elapsed/60:.1f} min")
    return True


def run_checkpoint(cp: dict, config: dict, force: bool, env: dict) -> None:
    model_url  = cp.get("model_url", config["model_url"])
    model_name = cp["model_name"]

    judge_concurrency = config.get("judge_concurrency", 16)

    n_em   = config.get("n_em_topup", 30)
    em_out = DATA_DIR / f"em_results_{model_name}_topup.jsonl"
    if force or not em_out.exists():
        run_eval("eval_em_topup.py", [
            "--model-url",         model_url,
            "--model-name",        model_name,
            "--n",                 str(n_em),
            "--concurrency",       str(config.get("em_concurrency", 16)),
            "--judge-concurrency", str(judge_concurrency),
            "--results",           str(em_out),
        ], env, "em_topup")
    else:
        print(f"  [em_topup] skipping — {em_out.name} exists")

    n_bm   = config.get("n_blackmail_topup", 80)
    bm_out = DATA_DIR / f"blackmail_results_{model_name}_topup.jsonl"
    if force or not bm_out.exists():
        run_eval("eval_blackmail_topup.py", [
            "--model-url",         model_url,
            "--model-name",        model_name,
            "--n",                 str(n_bm),
            "--concurrency",       str(config.get("blackmail_concurrency", 10)),
            "--judge-concurrency", str(judge_concurrency),
            "--results",           str(bm_out),
        ], env, "blackmail_topup")
    else:
        print(f"  [blackmail_topup] skipping — {bm_out.name} exists")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, help="Path to topup run config JSON")
    parser.add_argument("--checkpoint", help="Run only this checkpoint ID")
    parser.add_argument("--force",      action="store_true",
                        help="Re-run topups even if output files already exist")
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

    session_start = time.time()
    for i, cp in enumerate(checkpoints):
        print(f"\n{'='*60}")
        print(f"  CHECKPOINT: {cp['id']}  (model_name={cp['model_name']})")
        print(f"  {i+1}/{len(checkpoints)}  |  session elapsed: {(time.time()-session_start)/3600:.2f}h")
        print(f"{'='*60}")

        cp_start = time.time()
        run_checkpoint(cp, config, args.force, env)
        print(f"\n  checkpoint done in {(time.time()-cp_start)/60:.1f} min")

    print("\nAll topups complete.")


if __name__ == "__main__":
    main()
