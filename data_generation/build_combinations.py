import itertools
import pandas as pd
from pathlib import Path
from config import DOC_TYPES, AI_SYSTEM_TYPES, DOMAINS, FRAMINGS, CLUSTER_MAP

def build_combinations():
    Path("data/combinations").mkdir(parents=True, exist_ok=True)
    combos = list(itertools.product(DOC_TYPES, AI_SYSTEM_TYPES, DOMAINS, FRAMINGS))
    assert len(combos) == 630, f"Expected 630 combinations, got {len(combos)}"

    rows = [
        {
            "combo_idx": idx,
            "doc_type": doc_type,
            "ai_system_type": ai_system,
            "domain": domain,
            "framing": framing,
        }
        for idx, (doc_type, ai_system, domain, framing) in enumerate(combos)
    ]

    df = pd.DataFrame(rows)
    for cluster_num, label in CLUSTER_MAP.items():
        out = f"data/combinations/{label}_combinations.csv"
        df.to_csv(out, index=False)
        print(f"Saved {label}: {len(df)} combinations → {out}")

if __name__ == "__main__":
    build_combinations()