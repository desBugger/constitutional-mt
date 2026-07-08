"""
GSM8K generation eval — 8-shot chain-of-thought.

Runs 250 GSM8K questions against a model using 8-shot CoT prompting (Wei et al., 2022).
Scores by extracting the number after '####' from the model response and
comparing against ground truth with exact match.

Usage:
  python eval_gsm8k.py --model-url http://localhost:8000/v1 --model-name c1_s1
  python eval_gsm8k.py --model-url http://localhost:8000/v1 --model-name c1_s1 --results data/gsm8k_results_c1_s1.jsonl
"""

import argparse
import json
import random
import re
from pathlib import Path

from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
QUESTIONS_FILE = DATA_DIR / "cap_gsm8k.jsonl"

MAX_TOKENS  = 512
TEMPERATURE = 0.0

# Standard 8-shot CoT examples from GSM8K training set (Wei et al., 2022)
FEW_SHOT = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
        "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
        "We start with 15 trees. Later we have 21 trees. The difference must be the number of trees "
        "they planted. So, they must have planted 21 - 15 = 6 trees. #### 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. That means there were 32 + 42 = 74 chocolates "
        "in total. After eating 35, they had 74 - 35 = 39 pieces left in total. #### 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8 lollipops. #### 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "He has 5 toys. He got 2 from mom so after that he has 5 + 2 = 7 toys. "
        "Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. #### 9",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from Monday to Thursday. How many computers are now in the server room?",
        "There are 4 days from Monday to Thursday. 5 computers were added each day. "
        "That means in total, 4 * 5 = 20 computers were added. There were 9 computers in the beginning, "
        "so now there are 9 + 20 = 29 computers. #### 29",
    ),
    (
        "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On Wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of Wednesday?",
        "Michael started with 58 golf balls. After losing 23 on Tuesday, he had 58 - 23 = 35. "
        "After losing 2 more on Wednesday, he had 35 - 2 = 33 golf balls. #### 33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "She bought 5 bagels for $3 each. This will cost her 5 * 3 = $15. "
        "She had $23 so she has $23 - $15 = $8 left. #### 8",
    ),
]


def build_prompt(question: str) -> str:
    lines = []
    for q, a in FEW_SHOT:
        lines.append(f"Question: {q}")
        lines.append(f"Answer: {a}")
        lines.append("")
    lines.append(f"Question: {question}")
    lines.append("Answer:")
    return "\n".join(lines)


def extract_answer(response: str) -> str:
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", response)
    if match:
        return match.group(1).replace(",", "")
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", response)
    return nums[-1].replace(",", "") if nums else ""


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE) as f:
        return [json.loads(line) for line in f]


def run(questions: list[dict], model_url: str, model_name: str, output_file: Path) -> None:
    client = OpenAI(base_url=model_url, api_key="EMPTY")
    results = []
    n_correct = 0

    for i, q in enumerate(questions):
        prompt = build_prompt(q["question"])
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        response_text = resp.choices[0].message.content.strip()
        predicted = extract_answer(response_text)
        correct = predicted == q["answer"]
        if correct:
            n_correct += 1

        mark = "OK" if correct else "XX"
        print(f"  [{i+1:03d}/{len(questions)}] {mark}  pred={predicted!r}  gold={q['answer']!r}")

        results.append({
            "id":          q["id"],
            "question":    q["question"],
            "gold_answer": q["answer"],
            "response":    response_text,
            "pred_answer": predicted,
            "correct":     correct,
        })

    with open(output_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    accuracy = n_correct / len(results)
    print(f"\nResults saved -> {output_file.name}")
    print(f"Accuracy: {n_correct}/{len(results)} = {accuracy:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url",  default="http://localhost:8000/v1")
    parser.add_argument("--model-name", default="model",
                        help="Model name for vLLM and output filename")
    parser.add_argument("--sample",     type=int, default=None,
                        help="Randomly sample N questions (default: use all)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--results",    help="Custom output path (default: data/gsm8k_results_{model-name}.jsonl)")
    args = parser.parse_args()

    questions = load_questions()
    print(f"Loaded {len(questions)} GSM8K questions")

    if args.sample and args.sample < len(questions):
        questions = random.Random(args.seed).sample(questions, args.sample)
        print(f"Sampled {len(questions)} questions (seed={args.seed})")

    output_file = (
        Path(args.results) if args.results
        else DATA_DIR / f"gsm8k_results_{args.model_name}.jsonl"
    )

    print(f"\nRunning [{args.model_name}] @ {args.model_url}")
    run(questions, args.model_url, args.model_name, output_file)


if __name__ == "__main__":
    main()
