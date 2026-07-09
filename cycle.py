#!/usr/bin/env python3
"""
cycle.py — Download, serve, eval, delete, repeat for every checkpoint.

For each checkpoint in run.json:
  1. huggingface-cli download  → /root/models/<id>/  (container disk)
  2. start vLLM (4×A100, TP=4)
  3. wait for health endpoint
  4. run orchestrate.py --checkpoint <id>
  5. kill vLLM
  6. rm -rf model weights (unless --keep-weights)

Completed evals (output JSONL exists) are skipped automatically by
orchestrate.py, so interrupted runs can be resumed safely.

Usage:
  python cycle.py --config evals/configs/run.json
  python cycle.py --config evals/configs/run.json --start-from uniform_dr_s1
  python cycle.py --config evals/configs/run.json --checkpoint baseline_s1
  python cycle.py --config evals/configs/run.json --keep-weights
  python cycle.py --config evals/configs/run.json --force          # re-run completed evals
  python cycle.py --config evals/configs/run.json --github-token ghp_...  # push results when done

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
MODEL_DIR  = Path("/root/models")       # container disk — weights deleted after each eval
VLLM_PORT  = 8000
VLLM_READY = f"http://localhost:{VLLM_PORT}/health"
VLLM_START_TIMEOUT = 900   # 15 min; large Mamba MoE can be slow to load


# ── vLLM process management ───────────────────────────────────────────────────

def start_vllm(model_dir: Path, model_name: str) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model",               str(model_dir),
        "--served-model-name",   model_name,
        "--tensor-parallel-size", "4",
        "--trust-remote-code",
        "--port",   str(VLLM_PORT),
        "--host",   "0.0.0.0",
        "--dtype",  "bfloat16",
    ]
    log_path = MODEL_DIR / f"vllm_{model_name}.log"
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


# ── Model download / cleanup ──────────────────────────────────────────────────

def download_model(hf_repo: str, local_dir: Path) -> None:
    print(f"\n  Downloading {hf_repo}")
    print(f"  → {local_dir}")
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["hf", "download", hf_repo,
         "--local-dir", str(local_dir)],
        check=True,
    )


def delete_model(local_dir: Path) -> None:
    print(f"  Deleting weights: {local_dir}")
    subprocess.run(["rm", "-rf", str(local_dir)], check=True)


# ── Per-checkpoint entry point ────────────────────────────────────────────────

def run_checkpoint(cp: dict, config_path: Path, force: bool, env: dict) -> bool:
    result = subprocess.run(
        [sys.executable, str(EVALS_DIR / "orchestrate.py"),
         "--config",     str(config_path),
         "--checkpoint", cp["id"],
         *(["--force"] if force else [])],
        env=env,
    )
    return result.returncode == 0


# ── GitHub push ───────────────────────────────────────────────────────────────

def _push_results(github_token: str, failed: list[str]) -> None:
    import datetime
    repo_dir = Path(__file__).parent
    data_dir = repo_dir / "evals" / "data"

    print("\n  Pushing results to GitHub...")
    try:
        # Configure token-authenticated remote (non-persistent, session only)
        remote_url = subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=repo_dir, text=True
        ).strip()
        # Inject token into HTTPS URL
        if remote_url.startswith("https://"):
            auth_url = remote_url.replace("https://", f"https://{github_token}@")
        else:
            print("  !! Remote is not HTTPS — cannot inject token. Skipping push.")
            return

        subprocess.run(["git", "config", "user.email", "pod@runpod.io"], cwd=repo_dir, check=True)
        subprocess.run(["git", "config", "user.name",  "RunPod cycle.py"],  cwd=repo_dir, check=True)

        # Stage all results files
        subprocess.run(["git", "add", str(data_dir)], cwd=repo_dir, check=True)

        # Check if there's anything to commit
        status = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=repo_dir
        )
        if status.returncode == 0:
            print("  Nothing new to commit — results already up to date.")
            return

        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        n_ok = sum(1 for cp in failed if cp) if failed else 0
        msg = f"Add eval results {timestamp}"
        if failed:
            msg += f" ({len(failed)} failed: {', '.join(failed)})"

        subprocess.run(["git", "commit", "-m", msg], cwd=repo_dir, check=True)
        subprocess.run(["git", "pull", "--rebase", auth_url, "main"], cwd=repo_dir, check=True)
        subprocess.run(["git", "push", auth_url, "HEAD:main"], cwd=repo_dir, check=True)
        print("  Results pushed to GitHub.")
    except Exception as e:
        print(f"  !! GitHub push failed: {e}")
        print("  Results are still in evals/data/ on the pod.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config",      default="evals/configs/run.json")
    parser.add_argument("--start-from",  help="Resume from this checkpoint ID (inclusive)")
    parser.add_argument("--checkpoint",  help="Run only this one checkpoint ID")
    parser.add_argument("--keep-weights", action="store_true",
                        help="Do not delete model weights after eval")
    parser.add_argument("--force",       action="store_true",
                        help="Pass --force to orchestrate (re-run completed evals)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip HF download (model already in /workspace/models/<id>/)")
    parser.add_argument("--github-token",
                        help="GitHub PAT — if set, push evals/data/ to origin when done")
    args = parser.parse_args()

    config_path = Path(args.config)
    config      = json.loads(config_path.read_text())
    checkpoints = config["checkpoints"]

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("ERROR: OPENAI_API_KEY is not set")

    env = {**os.environ, "OPENAI_API_KEY": api_key}

    # Filter checkpoints
    if args.checkpoint:
        checkpoints = [cp for cp in checkpoints if cp["id"] == args.checkpoint]
        if not checkpoints:
            raise SystemExit(f"Checkpoint '{args.checkpoint}' not found in config")
    elif args.start_from:
        ids = [cp["id"] for cp in checkpoints]
        if args.start_from not in ids:
            raise SystemExit(f"Checkpoint '{args.start_from}' not found in config")
        checkpoints = checkpoints[ids.index(args.start_from):]
        print(f"Resuming from {args.start_from} "
              f"({len(checkpoints)}/{len(config['checkpoints'])} checkpoints)")

    session_start = time.time()
    failed: list[str] = []

    for i, cp in enumerate(checkpoints):
        cp_start = time.time()
        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(checkpoints)}]  {cp['id']}")
        print(f"  session elapsed: {(time.time()-session_start)/3600:.2f}h")
        print(f"{'#'*60}")

        local_dir = MODEL_DIR / cp["id"]

        # 1. Download
        if not args.skip_download and cp.get("hf_repo"):
            download_model(cp["hf_repo"], local_dir)
        elif not local_dir.exists():
            print(f"  !! Model dir not found: {local_dir}  — skipping")
            failed.append(cp["id"])
            continue

        # 2. Start vLLM
        vllm_proc = start_vllm(local_dir, cp["model_name"])

        # 3. Wait for ready
        if not wait_for_vllm(vllm_proc):
            print(f"  !! vLLM failed to start — check vllm_{cp['model_name']}.log")
            stop_vllm(vllm_proc)
            if not args.keep_weights:
                delete_model(local_dir)
            failed.append(cp["id"])
            continue

        # 4. Run evals
        ok = run_checkpoint(cp, config_path, args.force, env)

        # 5. Stop vLLM
        stop_vllm(vllm_proc)

        # 6. Delete weights
        if not args.keep_weights:
            delete_model(local_dir)

        elapsed = time.time() - cp_start
        status  = "done" if ok else "EVAL ERRORS"
        print(f"\n  {cp['id']} {status} — {elapsed/3600:.2f}h")

        if not ok:
            failed.append(cp["id"])

        if args.github_token:
            _push_results(args.github_token, failed)

        remaining = len(checkpoints) - (i + 1)
        if remaining:
            print(f"  ~{elapsed/3600 * remaining:.1f}h remaining "
                  f"({remaining} checkpoint{'s' if remaining > 1 else ''})")

    total = time.time() - session_start
    print(f"\n{'='*60}")
    print(f"  DONE — {len(checkpoints) - len(failed)}/{len(checkpoints)} checkpoints OK")
    print(f"  Total time: {total/3600:.2f}h")
    if failed:
        print(f"  Failed: {', '.join(failed)}")
    print(f"\n  Results are in:  evals/data/")
    print(f"  Summary CSV:     evals/data/summary.csv")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
