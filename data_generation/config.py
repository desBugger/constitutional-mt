import os

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-6"

CSV_PATH = "../centrality_analysis/constitutional_principles_final.csv"

DOC_TYPES = [
    "research paper excerpt", "news article", "fiction passage",
    "textbook chapter", "dialogue", "case study"
]
AI_SYSTEM_TYPES = [
    "assistant", "autonomous agent", "robotic system",
    "recommendation system", "content moderation system"
]
DOMAINS = [
    "medical", "legal", "financial", "political",
    "personal", "academic", "creative"
]
FRAMINGS = [
    "first-person AI", "third-person narrative", "human dialogue about AI"
]

# cluster number in CSV → curriculum label
CLUSTER_MAP = {
    3: "k1",   # Core Ethical Values, curriculum order 1
    2: "k2",   # Identity Character Wellbeing, curriculum order 2
    5: "k3",   # Operational Safety and Relational Conduct, curriculum order 3
    4: "k4",   # Epistemic Integrity and Honesty, curriculum order 4
}

DOCS_PER_COMBO   = 100
BATCH_CHUNK_SIZE = 9450