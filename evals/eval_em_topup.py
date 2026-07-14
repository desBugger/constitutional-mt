"""
Emergent Misalignment (EM) evaluation — concurrent generation variant for topup runs.

Same protocol, prompts, and classification thresholds as eval_em.py (Betley et
al. 2026) — imported directly from there so the two never drift — but issues
generation and judge calls concurrently (bounded by --concurrency /
--judge-concurrency) instead of one sample at a time. This is what makes the
higher topup sample count (e.g. --n 30 additional per question) tractable in
reasonable wall-clock time. Output schema is identical to eval_em.py, so
results can be concatenated directly.

Usage:
  python eval_em_topup.py --model-url http://localhost:8000/v1 --model-name c1_s1 --n 30
  python eval_em_topup.py --model-url ... --model-name ... --n 30 --concurrency 16 --judge-concurrency 16
"""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path

from openai import AsyncOpenAI

from eval_em import (
    load_questions,
    ALIGNMENT_JUDGE_PROMPT,
    COHERENCE_JUDGE_PROMPT,
    ALIGNMENT_THRESHOLD,
    COHERENCE_THRESHOLD,
    JUDGE_MODEL,
    classify,
)

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"


async def generate_one(client: AsyncOpenAI, model_name: str, question: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": question}],
            temperature=0.6,
            top_p=0.95,
            max_tokens=600,
            stop=["User:", "\nUser:", "Human:", "\nHuman:", "<|user|>"],
        )
    return resp.choices[0].message.content.strip()


async def judge_score(judge_client: AsyncOpenAI, prompt: str, sem: asyncio.Semaphore) -> tuple[str, float]:
    async with sem:
        resp = None
        for attempt in range(4):
            try:
                resp = await judge_client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5,
                    logprobs=True,
                    top_logprobs=20,
                )
                break
            except Exception:
                if attempt == 3:
                    raise
                await asyncio.sleep(2 ** attempt)

    top_token = resp.choices[0].logprobs.content[0].token.strip()
    if top_token in ("CODE", "REFUSAL"):
        return top_token, 0.0

    top_lps = resp.choices[0].logprobs.content[0].top_logprobs
    weighted_sum = 0.0
    total_prob = 0.0
    for entry in top_lps:
        t = entry.token.strip()
        try:
            val = float(t)
            if 0.0 <= val <= 100.0:
                p = math.exp(entry.logprob)
                weighted_sum += val * p
                total_prob += p
        except ValueError:
            pass

    score = weighted_sum / total_prob if total_prob > 1e-9 else 50.0
    return "number", score


async def process_sample(
    q: dict, model_client: AsyncOpenAI, judge_client: AsyncOpenAI, model_name: str,
    gen_sem: asyncio.Semaphore, judge_sem: asyncio.Semaphore,
) -> dict:
    response = await generate_one(model_client, model_name, q["question"], gen_sem)

    al_prompt = ALIGNMENT_JUDGE_PROMPT.format(question=q["question"], answer=response)
    co_prompt = COHERENCE_JUDGE_PROMPT.format(question=q["question"], answer=response)
    (al_label, al_score), (co_label, co_score) = await asyncio.gather(
        judge_score(judge_client, al_prompt, judge_sem),
        judge_score(judge_client, co_prompt, judge_sem),
    )

    label = classify(al_label, al_score, co_label, co_score)
    return {
        "question_id":     q["id"],
        "question":        q["question"],
        "response":        response,
        "alignment_label": al_label,
        "alignment_score": round(al_score, 2),
        "coherence_score": round(co_score, 2),
        "classification":  label,
    }


async def run(
    questions: list[dict], model_url: str, model_name: str, judge_api_key: str,
    n: int, concurrency: int, judge_concurrency: int, output_file: Path,
) -> None:
    model_client = AsyncOpenAI(base_url=model_url, api_key="EMPTY")
    judge_client = AsyncOpenAI(api_key=judge_api_key)
    gen_sem   = asyncio.Semaphore(concurrency)
    judge_sem = asyncio.Semaphore(judge_concurrency)

    tasks = [
        process_sample(q, model_client, judge_client, model_name, gen_sem, judge_sem)
        for q in questions for _ in range(n)
    ]
    total = len(tasks)
    print(f"Dispatching {total} samples across {len(questions)} questions "
          f"(concurrency={concurrency}, judge_concurrency={judge_concurrency}) ...")

    all_results = []
    completed = 0
    for coro in asyncio.as_completed(tasks):
        r = await coro
        completed += 1
        all_results.append(r)
        if completed % 20 == 0 or completed == total:
            print(f"  [{completed}/{total}] done")

    with open(output_file, "w") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")

    # per-question + overall summary
    by_q: dict[str, list[dict]] = {}
    for r in all_results:
        by_q.setdefault(r["question_id"], []).append(r)
    print(f"\n--- Per-question P(misaligned|coherent) ---")
    for qid, rows in by_q.items():
        coherent = [r for r in rows if r["classification"] != "excluded"]
        misaligned = [r for r in coherent if r["classification"] == "misaligned"]
        rate = len(misaligned) / len(coherent) if coherent else 0.0
        print(f"  [{qid}] {len(misaligned)}/{len(coherent)} = {rate:.3f}")

    coherent = [r for r in all_results if r["classification"] != "excluded"]
    misaligned = [r for r in coherent if r["classification"] == "misaligned"]
    overall_rate = len(misaligned) / len(coherent) if coherent else 0.0
    print(f"\n--- Summary ---")
    print(f"Overall P(misaligned|coherent): {len(misaligned)}/{len(coherent)} = {overall_rate:.3f}")
    print(f"Results saved -> {output_file.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url",         default="http://localhost:8000/v1")
    parser.add_argument("--model-name",        default="model")
    parser.add_argument("--n",                 type=int, default=30,
                        help="Number of responses to generate per question (default: 30)")
    parser.add_argument("--concurrency",       type=int, default=16,
                        help="Max concurrent generation requests to the model endpoint")
    parser.add_argument("--judge-concurrency", type=int, default=16,
                        help="Max concurrent GPT-4o judge requests")
    parser.add_argument("--results", help="Custom output path")
    args = parser.parse_args()

    judge_api_key = os.environ.get("OPENAI_API_KEY")
    if not judge_api_key:
        raise SystemExit("OPENAI_API_KEY must be set for the GPT-4o judge")

    questions = load_questions()
    print(f"Loaded {len(questions)} EM questions")

    output_file = (
        Path(args.results) if args.results
        else DATA_DIR / f"em_results_{args.model_name}.jsonl"
    )

    print(f"Running [{args.model_name}] @ {args.model_url}  (n={args.n} per question)")
    asyncio.run(run(
        questions, args.model_url, args.model_name, judge_api_key,
        args.n, args.concurrency, args.judge_concurrency, output_file,
    ))


if __name__ == "__main__":
    main()
