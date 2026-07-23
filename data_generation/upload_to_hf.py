import sys
from pathlib import Path
from huggingface_hub import HfApi

# Before running: huggingface-cli login
# Create your HuggingFace dataset repo first at hf.co

HF_REPO_ID = "cho-ai/constitutional-mt-data"

api = HfApi()

def upload_files(paths: list[Path]):
    print(f"Uploading {len(paths)} files to {HF_REPO_ID}...")
    for f in sorted(paths):
        print(f"  Uploading {f.name} ({f.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"data/{f.name}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        print(f"  Done: {f.name}")
    print("All done.")

def upload_final():
    """Upload all completed DR + standard files."""
    files = list(Path("data/output").glob("*_DR.jsonl")) + \
            list(Path("data/output").glob("*_noDR.jsonl"))
    upload_files(files)

def upload_pilots():
    """Upload all pilot files (checkpoint upload)."""
    files = list(Path("data/output").glob("*_pilot.jsonl"))
    upload_files(files)

def upload_specific(names: list[str]):
    """Upload specific named files from data/output/."""
    files = [Path("data/output") / n for n in names]
    missing = [f for f in files if not f.exists()]
    if missing:
        print(f"Missing files: {[f.name for f in missing]}")
        return
    upload_files(files)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "final"
    if mode == "final":
        upload_final()
    elif mode == "pilots":
        upload_pilots()
    elif mode == "specific":
        upload_specific(sys.argv[2:])
    else:
        print("Usage: python upload_to_hf.py [final|pilots|specific file1.jsonl file2.jsonl ...]")