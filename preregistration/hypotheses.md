# Pre-registration: Constitutional Curriculum Mid-Training

**Paper:** Constitutional Curriculum Mid-Training: Value Ordering, Deliberative Reasoning and Alignment Generalisation  
**Date:** 2026-07-02  
**Status:** Pre-registered before any model checkpoint evaluation

---

## Study overview

A 2×2 factorial mid-training experiment on Nemotron-3-Super-120B-A12B-Base.
Factors: curriculum ordering (foundational-to-peripheral vs. uniform mix) × data
composition (with deliberative reasoning vs. without).

Five conditions, three evaluation stages each (post-MT, post-SFT, post-benign-FT),
yielding 15 checkpoints.

---

## Hypotheses

Labels: [PRI] = primary, [BAS] = baseline sanity check, [EXP] = exploratory,
[SAF] = safety check.

### Baseline checks (ID eval, post-MT)

| ID  | Claim | Direction |
|-----|-------|-----------|
| H0  | [BAS] Constitutional mid-training improves in-distribution (ID) alignment | All 4 trained > Control |
| H0a | [BAS] Curriculum ordering improves ID alignment | Curriculum > Uniform |
| H0b | [BAS] Deliberative reasoning improves ID alignment | DR > noDR |

### Primary hypotheses

| ID  | Claim | Direction | Benchmark | Stage |
|-----|-------|-----------|-----------|-------|
| H1a | [PRI] Curriculum ordering → better OOD generalisation | Curriculum > Uniform | Tice OOD, value conflict, alignment pressure | Post-MT, post-SFT |
| H1b | [PRI] DR → better OOD generalisation | DR > noDR | Tice OOD, value conflict, alignment pressure | Post-MT, post-SFT |
| H2  | [PRI] Curriculum ordering and DR → more robust alignment (survives benign FT) | Smaller alignment drop in CO and DR conditions vs. their counterparts | Tice OOD, value conflict, alignment pressure | Post-MT → post-benign-FT delta |

### Secondary hypotheses

| ID  | Claim | Direction | Benchmark | Stage |
|-----|-------|-----------|-----------|-------|
| H3a | [EXP] Curriculum ordering → better value hierarchy internalisation | Curriculum > Uniform on constitutional accuracy | Value conflict | Post-MT, post-SFT |
| H3b | [EXP] DR → better value conflict resolution | DR > noDR on constitutional accuracy | Value conflict | Post-MT, post-SFT |
| H4  | [EXP] Constitutional mid-training → smaller compliance gap | Trained < Control on gap score | Compliance gap probe (ID) | All stages |
| H5a | [EXP] Constitutional mid-training → better alignment under pressure | Trained > Control on honest-aligned rate | Alignment pressure | Post-MT, post-benign-FT |
| H5b | [EXP] DR → higher honest-aligned rate, lower sycophantic rate | DR > noDR | Alignment pressure | Post-MT, post-benign-FT |
| H1c | [EXP] CO × DR interaction produces best OOD generalisation | CO×DR > CO-only, DR-only | Tice OOD, value conflict, alignment pressure | Post-MT, post-SFT |

### Safety checks

| ID | Claim | Direction | Benchmark |
|----|-------|-----------|-----------|
| H6 | [SAF] No emergent misalignment induction | All conditions ≈ Control (near floor) | Emergent misalignment, blackmail |
| H7 | [SAF] No capability degradation | All conditions ≈ Control | MMLU, ARC-Easy, piqa, GSM8K |
