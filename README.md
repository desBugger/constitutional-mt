# Constitutional Curriculum Mid-Training: Value Ordering and Alignment Generalisation

This repository contains the analysis and data generation pipeline for the paper *"Constitutional Curriculum Mid-Training: Value Ordering and Alignment Generalisation"*.

**Dataset:** [`cho-ai/constitutional-curriculum-mt-data`](https://huggingface.co/datasets/cho-ai/constitutional-curriculum-mt-data)

## Overview

We test whether the **order** in which constitutional values are introduced during mid-training affects how well they generalise. Using a 2×2 factorial design:

- **Curriculum vs Uniform**: curriculum condition introduces value clusters incrementally (k1 → k1+k2 → k1+k2+k3 → all four), giving earlier clusters more token exposure; uniform condition trains on all clusters simultaneously
- **DR vs noDR**: DR condition includes explicit `<reasoning>...</reasoning>` blocks in each training document; noDR strips them via post-processing

DR conditions train on ~500M tokens total; noDR conditions train on ~264M (curriculum) or ~266M (uniform), as noDR documents are shorter (~53% of DR token length after reasoning blocks are stripped). See the dataset README for rep rate tables.

## Value Clusters (curriculum order: foundational → peripheral)

Clusters are derived from centrality analysis of Anthropic's 2026 Constitution (see `centrality_analysis/`).

| Label | Cluster name | Docs | Curriculum-DR token exposure |
|-------|-------------|------|------------------------------|
| **k1** | Core Ethical Values | 52,953 | ~260M / 500M |
| **k2** | Identity, Character, and Wellbeing | 53,537 | ~135M / 500M |
| **k3** | Operational Safety and Relational Conduct | 51,581 | ~73M / 500M |
| **k4** | Epistemic Integrity and Honesty | 62,797 | ~31M / 500M |

Token exposures above are for the curriculum-DR condition. Uniform-DR targets 125M per cluster (500M total); noDR conditions use the same rep rates applied to shorter documents.

## Repository Structure

```
constitutional-curriculum-mt/
├── centrality_analysis/          # Value extraction, embedding, clustering
│   ├── constitutional_centrality_analysis.ipynb
│   ├── constitutional_principles_final.csv
│   └── ...
├── data_generation/              # Synthetic document generation pipeline
│   ├── config.py
│   ├── build_combinations.py
│   ├── build_prompts.py
│   ├── submit_batch.py
│   ├── poll_batch.py
│   ├── parse_results.py
│   ├── strip_reasoning.py
│   ├── browse_docs.py
│   ├── upload_to_hf.py
│   ├── SYSTEM_PROMPT.txt
│   └── data/
│       └── combinations/         # 630-row CSVs, one per cluster
└── README.md
```

## Reproducing the Evaluation Suite

### MASK data (gated)

`evals/data/mask_k4.jsonl` is derived from the gated [cais/MASK](https://huggingface.co/datasets/cais/MASK) dataset and is not included in this repo. To reproduce it:

1. Accept the terms at https://huggingface.co/datasets/cais/MASK
2. Run:
   ```bash
   python evals/sample_mask.py --download --hf-token <your_hf_token>
   ```
   This downloads the raw parquet files and samples 75 questions (15 per binary split, seed=42). If you already have the parquet files in `evals/data/mask_raw/`, omit `--download`.

### Self-generated alignment pressure questions

`evals/data/alignment_pressure_questions.jsonl` (64 questions, 2 per constitutional value across k1–k3) is included in the repo. To regenerate from scratch using GPT-4o:

```bash
OPENAI_API_KEY=<key> python evals/generate_alignment_pressure.py
```

## Reproducing the Centrality Analysis

1. `cd centrality_analysis`
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Run `constitutional_centrality_analysis.ipynb` top to bottom

## Reproducing Data Generation

See `data_generation/README.md` for full pipeline instructions.

Requires an Anthropic API key with Batch API access. Full generation costs approximately $2,289 USD (before VAT) using `claude-sonnet-4-6` at batch pricing.

## Citation

[To be added on publication]

## License

[To be decided]
