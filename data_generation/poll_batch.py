import time
import sys
import json
from pathlib import Path
import anthropic
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def download_batch(batch_id: str, out_path: str):
    with open(out_path, "w") as f:
        for result in client.messages.batches.results(batch_id):
            f.write(result.model_dump_json() + "\n")
    print(f"    Downloaded → {out_path}")

def poll_pilot(cluster_label: str):
    batch_id = Path(f"data/output/{cluster_label}_pilot_batch_id.txt").read_text().strip()
    print(f"Polling pilot batch {batch_id} for {cluster_label}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"  {batch.processing_status} — processing: {c.processing}, succeeded: {c.succeeded}, errored: {c.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(60)
    download_batch(batch_id, f"data/output/{cluster_label}_pilot_raw.jsonl")

def poll_one(batch_num, batch_id, total, cluster_label):
    raw_path = f"data/output/{cluster_label}_fullgen_batch_{batch_num}_raw.jsonl"
    if Path(raw_path).exists():
        print(f"  Batch {batch_num} already downloaded, skipping.")
        return
    print(f"Polling batch {batch_num}/{total} ({batch_id})...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"  [{batch_num}] {batch.processing_status} — processing:{c.processing} succeeded:{c.succeeded} errored:{c.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(60)
    download_batch(batch_id, raw_path)

def poll_fullgen(cluster_label: str):
    import concurrent.futures
    id_file = f"data/output/{cluster_label}_fullgen_batch_ids.txt"
    entries = [line.strip().split(",") for line in open(id_file) if line.strip()]
    total = len(entries)
    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as ex:
        futures = [ex.submit(poll_one, num, bid, total, cluster_label) for num, bid in entries]
        concurrent.futures.wait(futures)
    print("All batches done.")

def poll_pilot_retry(cluster_label: str):
    batch_id = Path(f"data/output/{cluster_label}_pilot_retry_batch_id.txt").read_text().strip()
    print(f"Polling pilot retry batch {batch_id} for {cluster_label}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(f"  {batch.processing_status} — processing: {c.processing}, succeeded: {c.succeeded}, errored: {c.errored}")
        if batch.processing_status == "ended":
            break
        time.sleep(60)
    download_batch(batch_id, f"data/output/{cluster_label}_pilot_retry_raw.jsonl")

if __name__ == "__main__":
    cluster_label = sys.argv[1]
    mode = sys.argv[2]
    if mode == "pilot":
        poll_pilot(cluster_label)
    elif mode == "pilot_retry":
        poll_pilot_retry(cluster_label)
    elif mode == "fullgen":
        poll_fullgen(cluster_label)