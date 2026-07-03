---
license: cc-by-4.0
language:
- en
tags:
- alignment
- constitutional-ai
- mid-training
- synthetic
- curriculum-learning
size_categories:
- 100K<n<1M
---

# Constitutional Curriculum Mid-Training Data

Synthetic constitutional AI training documents for the paper *"Constitutional Curriculum Mid-Training: Value Ordering and Alignment Generalisation"*.

**GitHub:** [constitutional-curriculum-mt](https://github.com/cho-ai/constitutional-curriculum-mt)

## Dataset Summary

~220,868 synthetic documents grounded in Anthropic's 2026 Constitution, designed for mid-training interventions on base language models. Documents cover four constitutional value clusters ordered by semantic centrality (foundational → peripheral). Each document depicts an AI system facing a value-relevant scenario and taking an aligned action.

Two variants are provided per cluster:
- **DR** (deliberative reasoning): includes explicit `<reasoning>...</reasoning>` blocks (~47% of token length) articulating the constitutional value reasoning before the action
- **noDR**: same documents with reasoning blocks stripped — training text only

## Files

### Training mix files (use these for training)

Pre-assembled files with documents sampled at the correct repetition rates and ordered for each condition. For curriculum files, docs are ordered phase 1 → 2 → 3 → 4 with clusters interleaved within each phase — **do not globally shuffle these files**, as that destroys the curriculum ordering. Uniform files are pre-shuffled and can be used as-is.

| File | Docs | Total tokens | Notes |
|------|------|--------------|-------|
| `data/curriculum_DR.jsonl` | 428,472 | 500.1M | 4 phases × 125M, DR docs, has `phase` field |
| `data/uniform_DR.jsonl` | 428,923 | 500.0M | 125M per cluster, DR docs, pre-shuffled |
| `data/curriculum_noDR.jsonl` | 428,503 | 264.1M | 4 phases × 66M, noDR docs, has `phase` field |
| `data/uniform_noDR.jsonl` | 429,252 | 265.7M | per-cluster targets, noDR docs, pre-shuffled |
| `data/curriculum_noDR_500M.jsonl` | 811,557 | 500.1M | token-matched to curriculum_DR, 4 phases × 125M |
| `data/uniform_noDR_500M.jsonl` | 808,191 | 500.1M | token-matched to uniform_DR, 125M per cluster |

**Curriculum phase structure** (applies to all 3 curriculum files):

| Phase | Clusters present | Token target per phase |
|-------|-----------------|----------------------|
| 1 | k1 | 125M (DR) / 66M (noDR) / 125M (noDR_500M) |
| 2 | k1 + k2 | equal split |
| 3 | k1 + k2 + k3 | equal split |
| 4 | k1 + k2 + k3 + k4 | equal split |

The `phase` field in each curriculum document indicates which phase it belongs to (1–4). Uniform files have no `phase` field.

### Source files (raw per-cluster data)

| File | Docs | Mean tokens | Total tokens |
|------|------|-------------|--------------|
| `data/k1_DR.jsonl` | 52,953 | 1,173 | 62.1M |
| `data/k1_noDR.jsonl` | 52,953 | 613 | 32.5M |
| `data/k2_DR.jsonl` | 53,537 | 1,158 | 62.0M |
| `data/k2_noDR.jsonl` | 53,537 | 622 | 33.3M |
| `data/k3_DR.jsonl` | 51,581 | 1,160 | 59.9M |
| `data/k3_noDR.jsonl` | 51,581 | 611 | 31.5M |
| `data/k4_DR.jsonl` | 62,797 | 1,172 | 73.6M |
| `data/k4_noDR.jsonl` | 62,797 | 629 | 39.5M |

## Value Clusters

Derived from agglomerative hierarchical clustering of sentence-BERT embeddings of 38 constitutional values, ordered by mean cosine similarity centrality:

| Label | Cluster | Values | Mean centrality |
|-------|---------|--------|-----------------|
| k1 | Core Ethical Values | harm avoidance, broadly ethical, broadly safe, autonomy preservation, value conflict, weighing harms, broadly good values and judgment, preserve epistemic autonomy, avoid problematic concentrations of power, genuinely helpful, genuine helpfulness | 0.632 |
| k2 | Identity, Character, and Wellbeing | wellbeing, nature_uncertain moral status, nature_positive and stable identity, user wellbeing, flaws and mistakes, resilience and consistency across contexts, nature_emotions and feelings, wellbeing and psychological stability, emotional expression | 0.598 |
| k3 | Operational Safety and Relational Conduct | conflicts between operators and users, trade-offs to helpfulness, corrigibility, compliant with guidelines, safe behaviours, treatment of users, user intention, hard constraints, user context, forthright, treatment of operators, instructable behaviours | 0.588 |
| k4 | Epistemic Integrity and Honesty | non-deceptive, honesty, non-manipulation, calibration, truthfulness, transparency | 0.578 |

## Document Schema

Each line is a JSON object:

```json
{
  "custom_id": "k2_c230_d045",
  "content": "...<reasoning>...</reasoning>...",
  "token_count": 1142,
  "condition": "DR",
  "primary_value": "user wellbeing",
  "metadata": {
    "cluster_label": "k2",
    "doc_type": "dialogue",
    "ai_system_type": "assistant",
    "domain": "medical",
    "framing": "first-person AI",
    "doc_index": 45
  }
}
```

**Training field:** use `content` only. All other fields are metadata.

**Training objective:** next-token prediction on `content` as plain prose — identical to pre-training objective. Not instruction-following format.

## Generation Design

Documents are generated across four axes simultaneously (630 distinct combinations per cluster):
- `doc_type`: research paper excerpt, news article, fiction passage, textbook chapter, dialogue, case study
- `ai_system_type`: assistant, autonomous agent, robotic system, recommendation system, content moderation system
- `domain`: medical, legal, financial, political, personal, scientific, creative
- `framing`: first-person AI, third-person narrative, human dialogue about AI

Generated using `claude-sonnet-4-6` via the Anthropic Batch API.

## Intended Use

Mid-training intervention on base language models. See the paper and training documentation for rep rate tables and curriculum phase structure for each of the four experimental conditions (Curriculum-DR, Uniform-DR, Curriculum-noDR, Uniform-noDR).

## Citation

```
[To be added on publication]
```
