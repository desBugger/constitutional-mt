# Constitutional Curriculum Mid-Training: Value Ordering and Alignment Generalisation

This repository contains the analysis and data generation pipeline for the paper *"Constitutional Curriculum Mid-Training: Value Ordering and Alignment Generalisation"*.

## Overview

We test whether the **order** in which constitutional values are introduced during mid-training affects how well they generalise. Using a 2×2 factorial design:

- **Curriculum vs Uniform**: curriculum condition introduces value clusters incrementally (k1 → k1+k2 → k1+k2+k3 → all four), giving earlier clusters more token exposure; uniform condition trains on all clusters simultaneously
- **DR vs noDR**: DR condition includes explicit `<reasoning>...</reasoning>` blocks in each training document; noDR strips them via post-processing

Each condition trains on ~500M tokens of synthetic constitutional AI documents generated via the Anthropic Batch API.

## Value Clusters (curriculum order: foundational → peripheral)

Clusters are derived from centrality analysis of Anthropic's 2026 Constitution (see `centrality_analysis/`).

| Label | Cluster name | Curriculum token exposure |
|-------|-------------|--------------------------|
| **k1** | Core Ethical Values | ~260M / 500M |
| **k2** | Identity, Character, and Wellbeing | ~135M / 500M |
| **k3** | Operational Safety and Relational Conduct | ~73M / 500M |
| **k4** | Epistemic Integrity and Honesty | ~31M / 500M |

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

## Reproducing the Centrality Analysis

1. `cd centrality_analysis`
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Run `constitutional_centrality_analysis.ipynb` top to bottom

## Reproducing Data Generation

See `data_generation/README.md` for full pipeline instructions.

Requires an Anthropic API key with Batch API access. Full generation costs approximately $2,230 USD (before VAT) using `claude-sonnet-4-6` at batch pricing.

## Citation

[To be added on publication]

## License

[To be decided]
