# Constitutional Midtraining: Centrality Analysis

This repository contains the constitutional value extraction and centrality analysis 
for the paper "Constitutional Curriculum Mid-Training: Value Ordering and Alignment Generalisation".

## Overview

We manually extract constitutional values from Anthropic's 2026 Constitution, embed 
them using SBERT (all-mpnet-base-v2), compute pairwise cosine similarity, and rank 
values by centrality to identify foundational versus peripheral constitutional values. 
We then apply agglomerative hierarchical clustering to identify curriculum units for 
constitutional mid-training.

## Curriculum Units (foundational → peripheral)

- **k1** — Core Ethical Values
- **k2** — Operational Safety and Relational Conduct
- **k3** — Identity, Character, and Wellbeing
- **k4** — Epistemic Integrity and Honesty
  
## Repository Structure

constitutional-embedding/
├── constitutional_centrality_analysis.ipynb  # Main analysis notebook
├── Manual Constitutional Value Extraction.xlsx  # Manually extracted values
├── requirements.txt  # Pinned dependencies
├── outputs/
│   ├── constitutional_embeddings.npy
│   ├── similarity_matrix.npy
│   ├── centrality_ranked.csv
│   ├── cluster_representatives.csv
│   └── constitutional_principles_final.csv
└── figures/
    ├── constitutional_pca_clusters.png
    └── constitutional_dendrogram.png

## Reproducing the Analysis

1. Clone the repository
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Open Jupyter Lab: `jupyter lab`
6. Run `constitutional_centrality_analysis.ipynb` from top to bottom

## Citation

[To be added on publication]

## License

[To be decided]
