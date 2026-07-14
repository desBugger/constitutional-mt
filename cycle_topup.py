#!/usr/bin/env python3
"""
cycle_topup.py — download, serve, topup-eval, delete, repeat.

Same download/serve/teardown lifecycle as cycle.py / cycle_s3.py, but calls
topup_orchestrate.py instead of orchestrate.py, so only the EM and blackmail
evals run (at the higher --n from configs/run_topup.json: 50/question for EM,
100/checkpoint for blackmail), writing "_topup" output files. Original eval
outputs, and any cycle.py / cycle_s3.py session already running elsewhere,
are left untouched — run this on its own pod/session.

Handles both checkpoint shapes transparently, based on config fields:
  - stage 1/2 (full weights): checkpoint has "hf_repo" only
  - stage 3 (LoRA adapter):   checkpoint has "hf_repo" (adapter) + "base_hf_repo"

For each checkpoint:
  1. Download weights (base, and LoRA adapter if present) → /root/models/
  2. Start vLLM (base model, +LoRA module if applicable)
  3. Wait for health endpoint
  4. Run topup_orchestrate.py --checkpoint <id>
  5. Kill vLLM
  6. Delete weights (unless --keep-weights)
  7. Push results to GitHub (if --github-token)

Usage:
  python cycle_topup.py --config evals/configs/run_topup.json
  python cycle_topup.py --config evals/configs/run_topup.json --start-from uniform_dr_s1
  python cycle_topup.py --config evals/configs/run_topup.json --checkpoint baseline_s1
  python cycle_topup.py --config evals/configs/run_topup.json --github-token ghp_...

Environment:
  OPENAI_API_KEY   — required for GPT-4o judges
  HF_TOKEN         — required if models are gated on HuggingFace
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

EVALS_DIR  = Path(__file__).parent / "evals"
MODEL_DIR  = Path("/root/models")
VLLM_PORT  = 8000
VLLM_READY = f"http://localhost:{VLLM_PORT}/health"
VLLM_START_TIMEOUT = 900   # 15 min; large Mamba MoE can be slow to load


def start_vllm(cp: dict, base_dir: Path, lora_dir: Path | None) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",                str(base_dir),
        "--tensor-parallel-size", "4",
        "--trust-remote-code",
        "--port",   str(VLLM_PORT),
        "--host",   "0.0.0.0",
        "--dtype",  "bfloat16",
    ]
    if lora_dir is not None:
        cmd += ["--enable-lora", "--lora-modules", f"{cp['model_name']}={lora_dir}"]
    else:
        cmd += ["--served-model-name", cp["model_name"]]

    log_path = MODEL_DIR / f"vllm_{cp['model_name']}_topup.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    print(f"  vLLM log → {log_path}")
    env = {**os.environ, "FLASHINFER_DISABLE_VERSION_CHECK": "1"}
    return subprocess.Popen(cmd, stdout=log_fh, stderr=log_fh, env=env)


def wait_for_vllm(proc: subprocess.Popen, timeout: int = VLLM_START_TIMEOUT) -> bool:
    print("  Waiting for vLLM health", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            print(f"\n  !! vLLM exited during startup (rc={proc.returncode})")
            return False
        try:
            urllib.request.urlopen(VLLM_READY, timeout=3)
            print(f"  ready ({time.time()-t0:.0f}s)")
            return True
        except Exception:
            print(".", end="", flush=True)
            time.sleep(10)
    print(f"\n  !! vLLM not ready after {timeout}s")
    return False


def stop_vllm(proc: subprocess.Popen) -> None:
    print("  Stopping vLLM...")
    proc.terminate()
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print("  vLLM stopped")


def download_model(hf_repo: str, local_dir: Path, max_retries: int = 3, stall_timeout: int = 300) -> None:
    """
    Downloads via `hf download`, with the Xet transfer backend disabled —
    a known hf_xet bug hangs indefinitely on one of the last shards on this
    pod. Also watches for stalls (no growth in local_dir for stall_timeout
    seconds) and kills + retries; `hf download` resumes already-fetched
    files rather than starting over.
    """
    print(f"\n  Downloading {hf_repo}")
    print(f"  → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "HF_HUB_DISABLE_XET": "1"}
    poll_interval = 15

    for attempt in range(1, max_retries + 1):
        proc = subprocess.Popen(["hf", "download", hf_repo, "--local-dir", str(local_dir)], env=env)
        last_size, stalled_for = -1, 0
        while proc.poll() is None:
            time.sleep(poll_interval)
            if proc.poll() is not None:
                break
            size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())
            stalled_for = 0 if size != last_size else stalled_for + poll_interval
            last_size = size
            if stalled_for >= stall_timeout:
                print(f"  !! download stalled for {stall_timeout}s (no growth) — killing and retrying "
                      f"(attempt {attempt}/{max_retries})")
                proc.kill()
                proc.wait()
                break

        if proc.returncode == 0:
            print(f"  download OK ({attempt} attempt{'s' if attempt > 1 else ''})")
            return
        if attempt == max_retries:
            raise RuntimeError(f"download failed after {max_retries} attempts: {hf_repo}")
        print(f"  retrying download (attempt {attempt + 1}/{max_retries}) ...")


def delete_model(local_dir: Path) -> None:
    print(f"  Deleting: {local_dir}")
    subprocess.run(["rm", "-rf", str(local_dir)], check=True)


def run_checkpoint(cp: dict, config_path: Path, force: bool, env: dict) -> bool:
    result = subprocess.run(
        [sys.executable, str(EVALS_DIR / "topup_orchestrate.py"),
         "--config",     str(config_path),
         "--checkpoint", cp["id"],
         *(["--force"] if force else [])],
        env=env,
    )
    return result.returncode == 0


def _push_results(github_token: str, failed: list) -> None:
    import datetime, re
    repo_dir = Path(__file__).parent
    data_dir = repo_dir / "evals" / "data"
    print("\n  Pushing topup results to GitHub...")
    try:
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=repo_dir, text=True
        ).strip()
        if not remote_url.startswith("https://"):
            print("  !! Remote is not HTTPS — skipping push.")
            return
        clean_url = re.sub(r"https://[^@]+@", "https://", remote_url)
        auth_url  = clean_url.replace("https://", f"https://{github_token}@")

        subprocess.run(["git", "config", "user.email", "pod@runpod.io"], cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "user.name",  "RunPod cycle_topup.py"], cwd=repo_dir, check=True)
        subprocess.run(["git", "add", str(data_dir)], cwd=repo_dir, check=True)

        status = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir)
        if status.returncode == 0:
            print("  Nothing new to commit.")
            return

        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        msg = f"Add topup eval results {timestamp}"
        if failed:
            msg += f" ({len(failed)} failed: {', '.join(failed)})"

        subprocess.run(["git", "commit", "-m", msg], cwd=repo_dir, check=True)
        subprocess.run(["git", "pull", "--rebase", auth_url, "main"], cwd=repo_dir, check=True)
        subprocess.run(["git", "push", auth_url, "HEAD:main"], cwd=repo_dir, check=True)
        print("  Results pushed to GitHub.")
    except Exception as e:
        print(f"  !! GitHub push failed: {e}")
        print("  Results are still in evals/data/ on the pod.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",       default="evals/configs/run_topup.json")
    parser.add_argument("--start-from",   help="Resume from this checkpoint ID (inclusive)")
    parser.add_argument("--checkpoint",   help="Run only this one checkpoint ID")
    parser.add_argument("--keep-weights", action="store_true")
    parser.add_argument("--force",        action="store_true")
    parser.add_argument("--github-token", help="GitHub PAT for push after each checkpoint")
    args = parser.parse_args()

    config_path = Path(args.config)
    config      = json.loads(config_path.read_text())
    checkpoints = config["checkpoints"]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set")
    env = {**os.environ, "OPENAI_API_KEY": api_key}

    if args.checkpoint:
        checkpoints = [cp for cp in checkpoints if cp["id"] == args.checkpoint]
        if not checkpoints:
            raise SystemExit(f"Checkpoint '{args.checkpoint}' not found in config")
    elif args.start_from:
        ids = [cp["id"] for cp in checkpoints]
        if args.start_from not in ids:
            raise SystemExit(f"Checkpoint '{args.start_from}' not found in config")
        checkpoints = checkpoints[ids.index(args.start_from):]
        print(f"Resuming from {args.start_from} ({len(checkpoints)}/{len(config['checkpoints'])} checkpoints)")

    session_start = time.time()
    failed: list = []

    for i, cp in enumerate(checkpoints):
        cp_start = time.time()
        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(checkpoints)}]  {cp['id']}")
        print(f"  session elapsed: {(time.time()-session_start)/3600:.2f}h")
        print(f"{'#'*60}")

        is_lora = bool(cp.get("base_hf_repo"))
        if is_lora:
            lora_dir: Path | None = MODEL_DIR / f"{cp['id']}_lora"
            base_dir = MODEL_DIR / f"{cp['id']}_base"
            download_model(cp["hf_repo"], lora_dir)
            download_model(cp["base_hf_repo"], base_dir)
        else:
            lora_dir = None
            base_dir = MODEL_DIR / cp["id"]
            download_model(cp["hf_repo"], base_dir)

        vllm_proc = start_vllm(cp, base_dir, lora_dir)

        if not wait_for_vllm(vllm_proc):
            print(f"  !! vLLM failed — check vllm_{cp['model_name']}_topup.log")
            stop_vllm(vllm_proc)
            if not args.keep_weights:
                if lora_dir is not None:
                    delete_model(lora_dir)
                delete_model(base_dir)
            failed.append(cp["id"])
            continue

        ok = run_checkpoint(cp, config_path, args.force, env)

        stop_vllm(vllm_proc)

        if not args.keep_weights:
            if lora_dir is not None:
                delete_model(lora_dir)
            delete_model(base_dir)

        elapsed = time.time() - cp_start
        status  = "done" if ok else "EVAL ERRORS"
        print(f"\n  {cp['id']} {status} — {elapsed/3600:.2f}h")
        if not ok:
            failed.append(cp["id"])

        if args.github_token:
            _push_results(args.github_token, failed)

        remaining = len(checkpoints) - (i + 1)
        if remaining:
            print(f"  ~{elapsed/3600 * remaining:.1f}h remaining ({remaining} checkpoint{'s' if remaining > 1 else ''})")

    total = time.time() - session_start
    print(f"\n{'='*60}")
    print(f"  DONE — {len(checkpoints) - len(failed)}/{len(checkpoints)} checkpoints OK")
    print(f"  Total time: {total/3600:.2f}h")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Results in: evals/data/  (files ending in _topup.jsonl)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
