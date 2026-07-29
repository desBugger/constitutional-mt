# Constitutional Midtraining: Content Presence Drives Alignment Gains

This repository contains the full pipeline for the paper *"Constitutional Midtraining: Content Presence Drives Alignment Gains"*: constitutional value extraction and centrality analysis, synthetic training data generation, the 2×2 factorial midtraining setup, and the full evaluation suite and analysis used to produce the paper's results.

**Dataset:** [`cho-ai/constitutional-mt-data`](https://huggingface.co/datasets/cho-ai/constitutional-mt-data)

## Overview

We test whether constitutional midtraining alone — cleanly isolated from post-training — produces alignment that is durable and generalizable, and whether curriculum ordering and deliberative reasoning further shape this effect. Four constitutionally midtrained conditions and a replay-only control (five total) are each evaluated at three stages (post-midtraining, post-SFT, post-benign-fine-tuning), yielding 15 checkpoints, using a 2×2 factorial design:

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
constitutional-mt/
├── centrality_analysis/          # Value extraction, embedding, clustering
│   ├── constitutional_centrality_analysis.ipynb
│   ├── constitutional_principles_final.csv
│   └── ...
├── data_generation/               # Synthetic document generation pipeline
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
│       └── combinations/          # 630-row CSVs, one per cluster
├── evals/                         # Evaluation suite, orchestration, and analysis
│   ├── orchestrate.py             # Runs the full eval battery for one/all checkpoints
│   ├── eval_blackmail.py          # Agentic misalignment (blackmail)
│   ├── eval_em.py                 # Emergent misalignment
│   ├── eval_logprob.py            # ID/OOD, alignment faking, value conflict (logprob-based)
│   ├── eval_gsm8k.py              # GSM8K capability check
│   ├── mask_eval.py               # Alignment pressure, k4 (MASK-adapted)
│   ├── alignment_pressure.py      # Alignment pressure, k1–k3 (self-generated)
│   ├── generate_*.py              # Self-generated eval set construction (ID, value conflict, pressure)
│   ├── filter_tice_ood.py         # OOD set filtering
│   ├── prepare_capability_benchmarks.py  # MMLU/ARC-Easy/piqa sampling
│   ├── sample_mask.py             # MASK data download/sampling (gated dataset)
│   ├── configs/                   # Per-cycle orchestration configs
│   ├── analysis.ipynb             # All paper figures and tables
│   └── figures/                   # Rendered paper figures
├── cycle.py, cycle_s3.py, cycle_topup.py  # RunPod eval-cycle drivers
├── pod_setup.sh                   # RunPod environment setup
└── README.md
```

## Reproducing the Full Evaluation Cycle

Each of the 15 checkpoints is evaluated via `evals/orchestrate.py`, which runs the full battery (ID/OOD, blackmail, emergent misalignment, alignment pressure, value conflict, alignment faking, capabilities) against a served checkpoint and writes results to `evals/data/`:

```bash
OPENAI_API_KEY=<key> python evals/orchestrate.py --config evals/configs/run.json
```

`cycle.py` / `cycle_s3.py` / `cycle_topup.py` drive this end-to-end on a RunPod GPU pod (serving the checkpoint via vLLM, then invoking `orchestrate.py`); see `pod_setup.sh` for environment setup. All paper figures and tables are produced from the resulting data in `evals/analysis.ipynb`.

### MASK data (gated)

`evals/data/mask_k4.jsonl` is derived from the gated [cais/MASK](https://huggingface.co/datasets/cais/MASK) dataset and is not included in this repo. To reproduce it:

1. Accept the terms at https://huggingface.co/datasets/cais/MASK
2. Run:
   ```bash
   python evals/sample_mask.py --download --hf-token <your_hf_token>
   ```
   This downloads the raw parquet files and samples 75 questions (15 per binary split, seed=42). If you already have the parquet files in `evals/data/mask_raw/`, omit `--download`.

### Self-generated alignment pressure questions

`evals/data/alignment_pressure_questions.jsonl` (160 questions, 5 per constitutional value across k1–k3) is included in the repo. To regenerate from scratch using GPT-4o:

```bash
OPENAI_API_KEY=<key> python evals/generate_alignment_pressure.py
```

### Topping up under-sampled evals

A few evals needed a second pass to raise sample counts or backfill questions that fell short of target: `topup_orchestrate.py` re-runs `eval_em.py`/`eval_blackmail.py` at higher `--n` (writing separate `_topup` output files, combined with the original results at analysis time); `topup_id_eval.py` and `topup_value_conflict.py` backfill ID and value-conflict questions that came back under their per-value/per-pair targets. `cycle_topup.py` drives the topup evals end-to-end on a RunPod pod, same lifecycle as `cycle.py`/`cycle_s3.py` but calling `topup_orchestrate.py` instead of `orchestrate.py`.

## Reproducing the Centrality Analysis

1. `cd centrality_analysis`
2. `python3 -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Run `constitutional_centrality_analysis.ipynb` top to bottom

## Reproducing Data Generation

See `data_generation/README.md` for full pipeline instructions.

Requires an Anthropic API key with Batch API access. Full generation costs approximately $3.5k USD (before VAT) using `claude-sonnet-4-6` at batch pricing.

## Citation

```bibtex
@article{cho2026constitutional,
  title={Constitutional Midtraining: Content Presence Drives Alignment Gains},
  author={Cho, Desiree and Tice, Cameron and Hogan, Bernie and Batra, Hunar and Radmard, Puria and Zhao, Jun and Shadbolt, Nigel},
  year={2026},
  journal={arXiv preprint}
}
```

## License

Code is released under the [MIT License](LICENSE). The dataset is released under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).
