import sys
from pathlib import Path
from huggingface_hub import HfApi

# Before running: huggingface-cli login
# Create your HuggingFace dataset repo first at hf.co
# e.g. https://huggingface.co/new-dataset

HF_REPO_ID = "PASTE_YOUR_HF_REPO_ID_HERE"  # e.g. "desirooo/constitutional-curriculum-mt-data"

api = HfApi()

def upload_dataset():
    files = list(Path("data/output").glob("*_DR.jsonl")) + \
            list(Path("data/output").glob("*_standard.jsonl"))

    print(f"Uploading {len(files)} files to {HF_REPO_ID}...")
    for f in files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"data/{f.name}",
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        print(f"  Uploaded: {f.name}")
    print("Done.")

if __name__ == "__main__":
    upload_dataset()