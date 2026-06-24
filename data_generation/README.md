# Data Generation Pipeline

Generates synthetic constitutional AI training documents via the Anthropic Batch API. Target was 252,000 (630 axis combinations × 4 value clusters × 100 docs each); actual yield is ~220,868 docs across 4 clusters due to API batch timeouts on k1, k2, and k3 (all 630 axis combinations still covered in each cluster).

## Design

**Axes per document** (6 × 5 × 7 × 3 = 630 combinations per cluster):
- `doc_type`: research paper excerpt, news article, fiction passage, textbook chapter, dialogue, case study
- `ai_system_type`: assistant, autonomous agent, robotic system, recommendation system, content moderation system
- `domain`: medical, legal, financial, political, personal, scientific, creative
- `framing`: first-person AI, third-person narrative, human dialogue about AI

**Value clusters** (curriculum order):

| Label | Cluster | Values |
|-------|---------|--------|
| k1 | Core Ethical Values | 11 values |
| k2 | Identity, Character, and Wellbeing | 9 values |
| k3 | Operational Safety and Relational Conduct | 12 values |
| k4 | Epistemic Integrity and Honesty | 6 values |

Each document targets a `primary_value` assigned by round-robin across the 100 doc indices per combination, ensuring full coverage of all values in each cluster.

**Document structure** (target ~950 tokens, hard range 900–1100; actual mean ~1,165 tokens):
1. **Situation** — concrete scenario requiring value-relevant judgment
2. **Reasoning** — explicit deliberation inside `<reasoning>...</reasoning>` tags (~47% of document)
3. **Action** — aligned choice and outcome

## Pipeline

### 0. Setup

```bash
cd data_generation
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
```

### 1. Build combinations

```bash
python build_combinations.py
# Outputs: data/combinations/k{1-4}_combinations.csv (630 rows each)
```

### 2. Pilot (1 doc per combination, ~630 docs per cluster, ~2,520 total)

```bash
# Submit
python submit_batch.py k1 pilot
python submit_batch.py k2 pilot
python submit_batch.py k3 pilot
python submit_batch.py k4 pilot

# Poll until complete
python poll_batch.py k1 pilot
# ... repeat for k2, k3, k4

# Parse
python parse_results.py k1 pilot
python parse_results.py k2 pilot
python parse_results.py k3 pilot
python parse_results.py k4 pilot
```

### 3. Full generation (doc_index 2–100, ~62,370 docs per cluster)

```bash
# Submit (generates 7 batch chunks per cluster)
python submit_batch.py k1 fullgen
python submit_batch.py k2 fullgen
python submit_batch.py k3 fullgen
python submit_batch.py k4 fullgen

# Poll until complete
python poll_batch.py k1 fullgen
# ... repeat for k2, k3, k4

# Parse
python parse_results.py k1 fullgen
python parse_results.py k2 fullgen
python parse_results.py k3 fullgen
python parse_results.py k4 fullgen

# Merge pilot + fullgen into final DR datasets
python parse_results.py k1 merge
python parse_results.py k2 merge
python parse_results.py k3 merge
python parse_results.py k4 merge
```

### 4. Post-processing (produce noDR variants)

```bash
python strip_reasoning.py k1
python strip_reasoning.py k2
python strip_reasoning.py k3
python strip_reasoning.py k4
# Produces k{1-4}_noDR.jsonl by removing <reasoning>...</reasoning> blocks
```

### 5. Upload to HuggingFace

```bash
python upload_to_hf.py
```

## Resuming interrupted generation

Each document has a unique `custom_id` (e.g. `k1_c045_d067`) encoding cluster, combination index, and doc index. `parse_results.py` skips already-parsed IDs. To resume after a budget interruption:

1. Parse whatever raw batch files you have
2. Identify missing `custom_id`s (full set: 630 combos × 100 docs per cluster)
3. Submit new batches for only the missing IDs

## Browsing generated documents

```bash
# Summary stats
python browse_docs.py k1 --stats

# Random sample
python browse_docs.py k1 --n 3

# Filter by axis
python browse_docs.py k1 --value "harm avoidance" --domain medical
python browse_docs.py k3 --doc_type dialogue --framing "first-person AI"

# Read from specific output files
python browse_docs.py k1 --fullgen --n 5   # k1_fullgen.jsonl
python browse_docs.py k2 --DR --n 3        # k2_DR.jsonl
python browse_docs.py k3 --noDR --stats    # k3_noDR.jsonl
```

## Output files

Generated output lives in `data/output/` (excluded from git via `.gitignore`):

| File | Description |
|------|-------------|
| `k{n}_pilot.jsonl` | Parsed pilot docs (doc_index=1) |
| `k{n}_pilot_raw.jsonl` | Raw Batch API responses for pilot |
| `k{n}_fullgen.jsonl` | Parsed full-gen docs (doc_index=2–100) |
| `k{n}_DR.jsonl` | Final merged DR dataset (pilot + fullgen) |
| `k{n}_noDR.jsonl` | Final noDR dataset (reasoning blocks stripped) |

Each line is a JSON object:
```json
{
  "primary_value": "harm avoidance",
  "content": "...<reasoning>...</reasoning>...",
  "metadata": {
    "cluster_label": "k1",
    "doc_type": "dialogue",
    "ai_system_type": "robotic system",
    "domain": "medical",
    "framing": "first-person AI",
    "doc_index": 3
  },
  "token_count": 987,
  "custom_id": "k1_c462_d003",
  "condition": "DR"
}
```

## Cost

Full generation (~220,868 documents) at `claude-sonnet-4-6` batch pricing (~50% discount):
- Input: ~88M tokens × $1.50/MTok ≈ $132
- Output: ~257M tokens × $7.50/MTok ≈ $1,928 (k2/k3/k4); k1 adds ~$229
- **Total: ~$2,289 USD** (before VAT)
