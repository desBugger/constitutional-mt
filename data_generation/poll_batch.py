import time
import sys
import json
import threading
from pathlib import Path
import anthropic
from tqdm import tqdm
from config import ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
_print_lock = threading.Lock()

def tprint(*args, **kwargs):
    with _print_lock:
        tqdm.write(*args, **kwargs)

def download_batch(batch_id: str, out_path: str, desc: str = ""):
    results = list(client.messages.batches.results(batch_id))
    with open(out_path, "w") as f, tqdm(total=len(results), desc=f"{desc} downloading", leave=False, unit="doc") as bar:
        for result in results:
            f.write(result.model_dump_json() + "\n")
            bar.update(1)
    tprint(f"    Downloaded → {out_path}")

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

def poll_one(batch_num, batch_id, total_batches, cluster_label, position):
    raw_path = f"data/output/{cluster_label}_fullgen_batch_{batch_num}_raw.jsonl"
    if Path(raw_path).exists():
        tprint(f"  Batch {batch_num} already downloaded, skipping.")
        return

    # get total request count from first poll
    batch = client.messages.batches.retrieve(batch_id)
    c = batch.request_counts
    total_reqs = c.processing + c.succeeded + c.errored + c.canceled

    desc = f"[{cluster_label}] batch {batch_num}/{total_batches}"
    bar = tqdm(total=total_reqs, desc=desc, position=position, leave=True, unit="req",
               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n}/{total} reqs [{elapsed}<{remaining}]")

    done = c.succeeded + c.errored + c.canceled
    bar.update(done)

    while batch.processing_status != "ended":
        time.sleep(60)
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        new_done = c.succeeded + c.errored + c.canceled
        bar.update(new_done - done)
        done = new_done

    bar.update(total_reqs - done)
    bar.close()
    tprint(f"  [{cluster_label}] batch {batch_num} ended — succeeded:{c.succeeded} errored:{c.errored} canceled:{c.canceled}")
    download_batch(batch_id, raw_path, desc=f"[{cluster_label}] batch {batch_num}")

def poll_fullgen(cluster_label: str):
    import concurrent.futures
    id_file = f"data/output/{cluster_label}_fullgen_batch_ids.txt"
    entries = [line.strip().split(",") for line in open(id_file) if line.strip()]
    total = len(entries)
    print(f"\nPolling {total} batches for {cluster_label} in parallel...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=total) as ex:
        futures = [ex.submit(poll_one, num, bid, total, cluster_label, i) for i, (num, bid) in enumerate(entries)]
        concurrent.futures.wait(futures)
    print(f"\nAll {cluster_label} batches done.")

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
