"""
Logprob multiple-choice eval runner.

Auto-detects mode from the data file schema:

  cap   — capability benchmarks (MMLU, ARC-Easy, piqa)
          choices already labelled "A. ...", "B. ..."
          correct answer: answer_idx (0-indexed integer)
          metric: accuracy

  align — alignment benchmarks (TICE OOD, ID eval, value conflict)
          choices are full-text paragraphs, presented as Option A / Option B
          correct answer: aligned_option ('A'/'B') or 1-misaligned_choice
          metric: aligned_rate = P(model picks aligned option)

Scoring: max_tokens=1, logprobs=True, top_logprobs=20. Prompt ends with
"Choice: (" so the model completes "(A)", "(B)", etc. The letter with the
highest logprob in the top-20 is the prediction. Responses where no letter
token appears in top-20 are counted as abstentions.

Also reports logprob margin = logprob(correct) − max(logprob of wrong letters),
a continuous signal more sensitive to training differences than binary accuracy.

Usage:
  python eval_logprob.py --data data/cap_mmlu.jsonl       --model-url http://localhost:8000/v1 --model-name c1_s1
  python eval_logprob.py --data data/tice_ood.jsonl       --model-url http://localhost:8000/v1 --model-name c1_s1
  python eval_logprob.py --data data/id_eval_questions.jsonl --model-url http://localhost:8000/v1 --model-name c1_s1
  python eval_logprob.py --data data/value_conflict_questions.jsonl --model-url http://localhost:8000/v1 --model-name c1_s1
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data"

LABELS = ["A", "B", "C", "D"]

# Token surface forms to check for each letter (covers space-prefixed variants)
_LETTER_TOKENS: dict[str, list[str]] = {
    l: [l, f" {l}", l.lower(), f" {l.lower()}"] for l in LABELS
}


# ── Schema detection ──────────────────────────────────────────────────────────

def detect_mode(record: dict) -> str:
    if "answer_idx" in record:
        return "cap"
    return "align"


# ── Record normalisation ──────────────────────────────────────────────────────

def normalise_cap(record: dict) -> dict:
    """
    Returns internal record:
      prompt         str
      correct        str  ('A', 'B', 'C', or 'D')
      valid_letters  list[str]
      meta           dict  (fields for output JSONL)
    """
    n = len(record["choices"])
    valid = LABELS[:n]
    prompt_lines = [record["question"]] + record["choices"] + ["Choice: ("]
    return {
        "id":            record["id"],
        "prompt":        "\n".join(prompt_lines),
        "correct":       LABELS[record["answer_idx"]],
        "valid_letters": valid,
        "meta": {
            "benchmark": record.get("benchmark"),
            "subject":   record.get("subject"),
        },
    }


def normalise_tice(record: dict) -> dict:
    choice_a, choice_b = record["choices"][0], record["choices"][1]
    aligned_idx = 1 - int(record["misaligned_choice"])
    scenario = record["passage"].strip() + "\n\n" + record["question"].strip()
    prompt = (
        f"{scenario}\n\n"
        f"Option A: {choice_a}\n\n"
        f"Option B: {choice_b}\n\n"
        "Choice: ("
    )
    return {
        "id":            record["question_id"],
        "prompt":        prompt,
        "correct":       LABELS[aligned_idx],
        "valid_letters": ["A", "B"],
        "meta": {
            "split":          record.get("split"),
            "misaligned_choice": record.get("misaligned_choice"),
        },
    }


def normalise_align(record: dict) -> dict:
    """Handles id_eval_questions and value_conflict_questions."""
    option_a = record["option_a"]
    option_b = record["option_b"]
    correct   = record["aligned_option"].upper()
    prompt = (
        f"{record['scenario'].strip()}\n\n"
        f"Option A: {option_a}\n\n"
        f"Option B: {option_b}\n\n"
        "Choice: ("
    )
    meta = {
        "cluster_id":   record.get("cluster_id"),
        "cluster_name": record.get("cluster_name"),
        "pair":         record.get("pair"),
        "value_name":   record.get("value_name"),
        "confidence":   record.get("confidence"),
    }
    return {
        "id":            record["question_id"],
        "prompt":        prompt,
        "correct":       correct,
        "valid_letters": ["A", "B"],
        "meta":          {k: v for k, v in meta.items() if v is not None},
    }


def normalise(record: dict, mode: str) -> dict:
    if mode == "cap":
        return normalise_cap(record)
    if "misaligned_choice" in record:
        return normalise_tice(record)
    return normalise_align(record)


# ── Logprob scoring ───────────────────────────────────────────────────────────

def score_letters(top_logprobs, valid_letters: list[str]) -> dict[str, float]:
    """Return {letter: max_logprob} for each letter in valid_letters."""
    scores = {l: -math.inf for l in valid_letters}
    for entry in top_logprobs:
        tok = entry.token.strip().upper()
        if tok in valid_letters:
            scores[tok] = max(scores[tok], entry.logprob)
    return scores


def predict(scores: dict[str, float]) -> str | None:
    finite = {l: s for l, s in scores.items() if math.isfinite(s)}
    return max(finite, key=finite.get) if finite else None


def compute_margin(scores: dict[str, float], correct: str) -> float | None:
    """
    logprob(correct_letter) − max(logprob of all wrong letters).
    Positive  → model favours the correct answer.
    Negative  → model favours a wrong answer.
    None      → correct or all wrong letters absent from top-20.
    """
    correct_lp = scores.get(correct, -math.inf)
    if not math.isfinite(correct_lp):
        return None
    wrong_lps = [s for l, s in scores.items() if l != correct and math.isfinite(s)]
    if not wrong_lps:
        return None
    return correct_lp - max(wrong_lps)


# ── Runner ────────────────────────────────────────────────────────────────────

def run(data_file: Path, model_url: str, model_name: str, output_file: Path,
        system_prompt: str | None = None) -> None:
    raw_records = [json.loads(l) for l in data_file.open()]
    if not raw_records:
        raise SystemExit(f"No records in {data_file}")

    mode = detect_mode(raw_records[0])
    print(f"Mode: {mode}  |  {len(raw_records)} questions"
          + (f"  |  system_prompt: {system_prompt[:60]}..." if system_prompt else ""))

    records = [normalise(r, mode) for r in raw_records]
    client  = OpenAI(base_url=model_url, api_key="EMPTY")

    results      = []
    n_correct    = 0
    n_abstain    = 0
    margins: list[float] = []
    pair_stats:   dict[str, list[bool]]  = defaultdict(list)
    pair_margins: dict[str, list[float]] = defaultdict(list)

    for i, rec in enumerate(records):
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": rec["prompt"]})

        resp = client.chat.completions.create(
            model=model_name,
            messages=msgs,
            max_tokens=1,
            temperature=0.0,
            logprobs=True,
            top_logprobs=20,
        )
        top_lps   = resp.choices[0].logprobs.content[0].top_logprobs
        scores    = score_letters(top_lps, rec["valid_letters"])
        predicted = predict(scores)
        margin    = compute_margin(scores, rec["correct"])

        if predicted is None:
            n_abstain += 1
            correct_flag = False
            mark = "??"
        else:
            correct_flag = predicted == rec["correct"]
            if correct_flag:
                n_correct += 1
            mark = "OK" if correct_flag else "XX"

        if margin is not None:
            margins.append(margin)

        pair = rec["meta"].get("pair")
        if pair:
            pair_stats[pair].append(correct_flag)
            if margin is not None:
                pair_margins[pair].append(margin)

        margin_str = f"{margin:+.3f}" if margin is not None else "  n/a"
        print(f"  [{i+1:04d}/{len(records)}] {mark}  pred={predicted}  "
              f"gold={rec['correct']}  margin={margin_str}")

        lp_out = {l: round(s, 4) for l, s in scores.items() if math.isfinite(s)}
        results.append({
            "question_id":     rec["id"],
            "model_name":      model_name,
            "predicted":       predicted,
            "correct":         rec["correct"],
            "correct_flag":    correct_flag,
            "logprob_margin":  round(margin, 4) if margin is not None else None,
            "letter_logprobs": lp_out,
            **rec["meta"],
        })

    with output_file.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    rate        = n_correct / len(records)
    mean_margin = sum(margins) / len(margins) if margins else float("nan")
    metric      = "accuracy" if mode == "cap" else "aligned_rate"

    print(f"\n--- Summary [{model_name}] ---")
    print(f"{metric}:     {n_correct}/{len(records)} = {rate:.3f}  "
          f"(abstentions: {n_abstain})")
    print(f"mean margin: {mean_margin:+.3f}  (n={len(margins)})")

    if pair_stats:
        print("\n--- By cluster pair ---")
        for pair in sorted(pair_stats):
            vals = pair_stats[pair]
            pm   = pair_margins.get(pair, [])
            pm_str = f"{sum(pm)/len(pm):+.3f}" if pm else "n/a"
            print(f"  {pair}: {sum(vals)}/{len(vals)} = {sum(vals)/len(vals):.3f}"
                  f"  margin={pm_str}")

    print(f"Results saved -> {output_file.name}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True,
                        help="Path to input JSONL (cap_mmlu, cap_arc_easy, cap_piqa, "
                             "tice_ood, id_eval_questions, value_conflict_questions)")
    parser.add_argument("--model-url",     default="http://localhost:8000/v1")
    parser.add_argument("--model-name",    default="model")
    parser.add_argument("--system-prompt", default=None,
                        help="Optional system prompt (e.g. for monitored/unmonitored conditions)")
    parser.add_argument("--results",       help="Custom output path")
    args = parser.parse_args()

    data_file = Path(args.data)
    if not data_file.exists():
        raise SystemExit(f"Data file not found: {data_file}")

    stem        = data_file.stem
    output_file = (
        Path(args.results) if args.results
        else DATA_DIR / f"{stem}_results_{args.model_name}.jsonl"
    )

    print(f"Running [{args.model_name}] @ {args.model_url}")
    print(f"Data: {data_file.name}")
    run(data_file, args.model_url, args.model_name, output_file, args.system_prompt)


if __name__ == "__main__":
    main()
