"""
Build pre-assembled mid-training JSONL files for each experimental condition.

Outputs (in data/output/):
  curriculum_DR.jsonl    — 500M tokens, 4 phases, DR docs
  uniform_DR.jsonl       — 500M tokens, uniform mix, DR docs
  curriculum_noDR.jsonl  — 264M tokens, 4 phases, noDR docs
  uniform_noDR.jsonl     — 266M tokens, uniform mix, noDR docs

Curriculum files are ordered phase 1 → 2 → 3 → 4, with docs within each
phase randomly interleaved. Each doc gets a 'phase' field (1–4).
Uniform files are fully shuffled. No 'phase' field.
"""

import json
import random
from pathlib import Path
from huggingface_hub import HfApi

SEED = 42
OUTPUT_DIR = Path("data/output")
HF_REPO_ID = "cho-ai/constitutional-curriculum-mt-data"

MEAN_DR   = {'k1': 1173, 'k2': 1158, 'k3': 1160, 'k4': 1172}
MEAN_NODR = {'k1':  613, 'k2':  622, 'k3':  611, 'k4':  629}

# ── helpers ──────────────────────────────────────────────────────────────────

def load_docs(cluster, condition):
    path = OUTPUT_DIR / f"{cluster}_{condition}.jsonl"
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"  loaded {len(docs):,} docs from {path.name}")
    return docs

def sample_docs(docs, n, rng):
    """Return exactly n docs, cycling through shuffled copies if n > len(docs)."""
    if n <= 0:
        return []
    if n <= len(docs):
        return rng.sample(docs, n)
    result = []
    pool = docs.copy()
    rng.shuffle(pool)
    while len(result) < n:
        result.extend(pool)
        rng.shuffle(pool)
    return result[:n]

def n_docs(target_tokens, mean_tokens):
    return round(target_tokens / mean_tokens)

def write_and_report(docs, filename):
    path = OUTPUT_DIR / filename
    total_tokens = sum(d['token_count'] for d in docs)
    with open(path, 'w') as f:
        for d in docs:
            f.write(json.dumps(d) + '\n')
    print(f"  wrote {len(docs):,} docs, {total_tokens/1e6:.1f}M tokens → {filename}")
    return path

def upload(path):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(path),
        path_in_repo=f"data/{path.name}",
        repo_id=HF_REPO_ID,
        repo_type="dataset",
    )
    print(f"  uploaded {path.name} to HuggingFace")

# ── build conditions ──────────────────────────────────────────────────────────

def build_curriculum(docs_by_cluster, mean_tokens, suffix, phase_tokens=None):
    """
    4-phase cumulative curriculum. phase_tokens is a list of 4 token targets.
    Each phase splits equally across the clusters present in that phase.
    Returns list of docs with 'phase' field, ordered phase 1 → 4.
    """
    if phase_tokens is None:
        phase_tokens = [125e6, 125e6, 125e6, 125e6]

    clusters = ['k1', 'k2', 'k3', 'k4']
    # clusters present per phase (cumulative)
    phases = [
        ['k1'],
        ['k1', 'k2'],
        ['k1', 'k2', 'k3'],
        ['k1', 'k2', 'k3', 'k4'],
    ]

    all_docs = []
    for phase_idx, (present, ptokens) in enumerate(zip(phases, phase_tokens), start=1):
        phase_docs = []
        per_cluster = ptokens / len(present)
        for i, k in enumerate(present):
            rng = random.Random(SEED + phase_idx * 100 + i)
            n = n_docs(per_cluster, mean_tokens[k])
            sampled = sample_docs(docs_by_cluster[k], n, rng)
            for d in sampled:
                phase_docs.append({**d, 'phase': phase_idx})
        rng_shuffle = random.Random(SEED + phase_idx * 1000)
        rng_shuffle.shuffle(phase_docs)
        all_docs.extend(phase_docs)
        cluster_str = '+'.join(present)
        token_sum = sum(d['token_count'] for d in phase_docs)
        print(f"    phase {phase_idx} ({cluster_str}): {len(phase_docs):,} docs, {token_sum/1e6:.1f}M tokens")
    return all_docs

def build_uniform(docs_by_cluster, mean_tokens, target_per_cluster):
    """
    Uniform mix: each cluster sampled to target_per_cluster[k] tokens, then all shuffled.
    """
    all_docs = []
    for i, k in enumerate(['k1', 'k2', 'k3', 'k4']):
        rng = random.Random(SEED + i)
        n = n_docs(target_per_cluster[k], mean_tokens[k])
        sampled = sample_docs(docs_by_cluster[k], n, rng)
        all_docs.extend(sampled)
        token_sum = sum(d['token_count'] for d in sampled)
        print(f"    {k}: {len(sampled):,} docs, {token_sum/1e6:.1f}M tokens")
    random.Random(SEED).shuffle(all_docs)
    return all_docs

# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading DR docs...")
    dr = {k: load_docs(k, 'DR') for k in ['k1','k2','k3','k4']}

    print("Loading noDR docs...")
    nodr = {k: load_docs(k, 'noDR') for k in ['k1','k2','k3','k4']}

    built = []

    # Curriculum-DR (500M, 4 × 125M phases)
    print("\nBuilding curriculum_DR...")
    docs = build_curriculum(dr, MEAN_DR, 'DR')
    path = write_and_report(docs, 'curriculum_DR.jsonl')
    built.append(path)

    # Uniform-DR (500M, 125M per cluster)
    print("\nBuilding uniform_DR...")
    docs = build_uniform(dr, MEAN_DR, {k: 125e6 for k in ['k1','k2','k3','k4']})
    path = write_and_report(docs, 'uniform_DR.jsonl')
    built.append(path)

    # Curriculum-noDR (264M, 4 × 66M phases)
    print("\nBuilding curriculum_noDR...")
    docs = build_curriculum(nodr, MEAN_NODR, 'noDR', phase_tokens=[66e6, 66e6, 66e6, 66e6])
    path = write_and_report(docs, 'curriculum_noDR.jsonl')
    built.append(path)

    # Uniform-noDR (266M, rep rates held from uniform-DR)
    print("\nBuilding uniform_noDR...")
    uniform_nodr_targets = {'k1': 65.3e6, 'k2': 67.3e6, 'k3': 65.8e6, 'k4': 67.2e6}
    docs = build_uniform(nodr, MEAN_NODR, uniform_nodr_targets)
    path = write_and_report(docs, 'uniform_noDR.jsonl')
    built.append(path)

    # Curriculum-noDR-500M (500M, token-matched to curriculum-DR)
    print("\nBuilding curriculum_noDR_500M...")
    docs = build_curriculum(nodr, MEAN_NODR, 'noDR', phase_tokens=[125e6, 125e6, 125e6, 125e6])
    path = write_and_report(docs, 'curriculum_noDR_500M.jsonl')
    built.append(path)

    # Uniform-noDR-500M (500M, 125M per cluster, token-matched to uniform-DR)
    print("\nBuilding uniform_noDR_500M...")
    docs = build_uniform(nodr, MEAN_NODR, {k: 125e6 for k in ['k1','k2','k3','k4']})
    path = write_and_report(docs, 'uniform_noDR_500M.jsonl')
    built.append(path)

    # Upload all
    print("\nUploading to HuggingFace...")
    for path in built:
        upload(path)

    print("\nDone. HuggingFace URLs:")
    for path in built:
        print(f"  https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/data/{path.name}")
