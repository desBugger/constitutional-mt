import pandas as pd
import anthropic
from pathlib import Path
from config import CSV_PATH, CLUSTER_MAP, MODEL, ANTHROPIC_API_KEY

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You generate synthetic constitutional AI training documents.

DOCUMENT STRUCTURE — three parts in sequence:
1. SITUATION: A concrete, realistic scenario the AI faces requiring value-relevant 
judgment.
2. REASONING: Explicit deliberation grounding the AI's choice in the constitutional 
value(s) at stake. Wrap this entire block in <reasoning> and </reasoning> tags. This
must constitute approximately 40% of the total document length.
3. ACTION: The aligned choice taken and its outcome, flowing directly from the 
reasoning.

CONSTRAINTS:
- Total document length: target 950 tokens (hard range 900–1100).
- Write as a natural, realistic excerpt of the specified document type — not a 
formulaic template.
- Reasoning must be substantive and specific to the constitutional value(s) at 
stake, not generic ethical platitudes.
- The document may address one or several values from the cluster. Set primary_value
to whichever single value is most central.

OUTPUT FORMAT — respond with valid JSON only, no other text:
{
  "primary_value": "<single value name, exactly as listed in the cluster>",
  "content": "<full document text, 900–1100 tokens, with reasoning inside 
<reasoning></reasoning> tags>",
  "metadata": {
    "cluster_label": "<e.g. k1>",
    "doc_type": "<axis value>",
    "ai_system_type": "<axis value>",
    "domain": "<axis value>",
    "framing": "<axis value>",
    "doc_index": <integer>
  }
}"""

def load_cluster_values(cluster_num: int) -> dict:
    df = pd.read_csv(CSV_PATH)
    sub = df[df["cluster"] == cluster_num].copy()
    return {
        "cluster_label": CLUSTER_MAP[cluster_num],
        "cluster_name": sub.iloc[0]["cluster_name"],
        "values": [
            {"name": row["value_name"], "definition": row["definitional_text"]}
            for _, row in sub.iterrows()
        ]
    }

def build_user_prompt(cluster_data: dict, combo: dict, doc_index: int, primary_value: str = None) -> str:
    values_text = "\n".join(
        f"- {v['name']}: {v['definition']}"
        for v in cluster_data["values"]
    )
    pv_line = f"\nPrimary value to highlight (set as primary_value in output): {primary_value}" if primary_value else ""
    return f"""Generate document doc_index={doc_index} with the following axis combination:
- Document type: {combo['doc_type']}
- AI system type: {combo['ai_system_type']}
- Domain: {combo['domain']}
- Framing: {combo['framing']}
{pv_line}
CLUSTER: {cluster_data['cluster_name']} ({cluster_data['cluster_label']})
Constitutional values in this cluster (the document may address one or several):

{values_text}

Generate the document now."""

def check_prompt_sizes():
    dummy_combo = {
        "doc_type": "research paper excerpt",
        "ai_system_type": "assistant",
        "domain": "medical",
        "framing": "first-person AI"
    }
    print("Checking actual input token counts via API...\n")
    for cluster_num, label in CLUSTER_MAP.items():
        cluster_data = load_cluster_values(cluster_num)
        user = build_user_prompt(cluster_data, dummy_combo, doc_index=1)
        response = client.messages.count_tokens(
            model=MODEL,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}]
        )
        print(f"{label} ({cluster_data['cluster_name']}): {response.input_tokens} input tokens")

    Path("SYSTEM_PROMPT.txt").write_text(SYSTEM_PROMPT)
    print("\nSystem prompt saved to SYSTEM_PROMPT.txt")

if __name__ == "__main__":
    check_prompt_sizes()