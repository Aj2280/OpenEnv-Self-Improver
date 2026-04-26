import os
import re
import random
from collections import Counter

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./math_grpo_mps_final"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VERBOSE = os.getenv("EVAL_VERBOSE", "").lower() in ("1", "true", "yes")
NUM_PROBLEMS = int(os.getenv("EVAL_NUM", "100"))
SELF_CONSISTENCY = int(os.getenv("EVAL_SC", "3"))  # extra sampled rollouts per problem


def build_test_cases(n: int = 100, seed: int = 0) -> list[tuple[str, float]]:
    """Deterministic suite of numeric problems with known answers."""
    rng = random.Random(seed)
    cases: list[tuple[str, float]] = []

    while len(cases) < n:
        kind = rng.randint(0, 7)
        if kind == 0:
            a, b = rng.randint(1, 99), rng.randint(1, 99)
            cases.append((f"{a} + {b}", float(a + b)))
        elif kind == 1:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            cases.append((f"{a} + {b}", float(a + b)))
        elif kind == 2:
            a = rng.randint(10, 99)
            b = rng.randint(1, a)
            cases.append((f"{a} - {b}", float(a - b)))
        elif kind == 3:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
            cases.append((f"{a} * {b}", float(a * b)))
        elif kind == 4:
            a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(1, 9)
            cases.append((f"{a} * ({b} + {c})", float(a * (b + c))))
        elif kind == 5:
            a, b, c = rng.randint(2, 8), rng.randint(2, 8), rng.randint(1, 12)
            cases.append((f"{a} * {b} + {c}", float(a * b + c)))
        elif kind == 6:
            x = rng.randint(2, 20)
            cases.append((f"sqrt({x * x})", float(x)))
        else:
            b = rng.randint(2, 9)
            a = rng.randint(3, 20)
            prod = a * b
            cases.append((f"{prod} / {b}", float(a)))

    return cases[:n]


if VERBOSE:
    print(f"Loading model from {MODEL_PATH} on {DEVICE}...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16 if DEVICE == "mps" else torch.float32,
).to(DEVICE)
model.eval()


def _build_prompt(problem_str: str) -> str:
    # Chain-of-thought few-shots in the GRPO `<thought>` format. The examples
    # explicitly demonstrate the operations the base model gets wrong:
    #   * two-digit + two-digit with carry
    #   * `a * 10` style multiplications
    #   * order-of-operations (parentheses first, multiply before add)
    #   * larger perfect squares for sqrt
    return (
        "You are a careful math tutor. Solve every problem step by step inside <thought> tags, "
        "then write the final integer on a new line after 'Answer:'.\n"
        "Problem: 4 + 9\n<thought>4+9=13.</thought>\nAnswer: 13\n"
        "Problem: 34 + 66\n<thought>4+6=10, write 0 carry 1. 3+6+1=10. Result: 100.</thought>\nAnswer: 100\n"
        "Problem: 79 + 85\n<thought>9+5=14, write 4 carry 1. 7+8+1=16. Result: 164.</thought>\nAnswer: 164\n"
        "Problem: 56 + 87\n<thought>6+7=13, write 3 carry 1. 5+8+1=14. Result: 143.</thought>\nAnswer: 143\n"
        "Problem: 46 - 9\n<thought>46-9=37.</thought>\nAnswer: 37\n"
        "Problem: 10 * 9\n<thought>10*9=90.</thought>\nAnswer: 90\n"
        "Problem: 11 * 5\n<thought>11*5=55.</thought>\nAnswer: 55\n"
        "Problem: 7 * 8 + 2\n<thought>Multiply first: 7*8=56. Then 56+2=58.</thought>\nAnswer: 58\n"
        "Problem: 6 * 3 + 5\n<thought>6*3=18. 18+5=23.</thought>\nAnswer: 23\n"
        "Problem: 6 * (3 + 9)\n<thought>Parentheses first: 3+9=12. Then 6*12=72.</thought>\nAnswer: 72\n"
        "Problem: 9 * (3 + 7)\n<thought>3+7=10. 9*10=90.</thought>\nAnswer: 90\n"
        "Problem: 9 * (3 + 2)\n<thought>3+2=5. 9*5=45.</thought>\nAnswer: 45\n"
        "Problem: sqrt(81)\n<thought>9*9=81, so sqrt(81)=9.</thought>\nAnswer: 9\n"
        "Problem: sqrt(324)\n<thought>18*18=324, so sqrt(324)=18.</thought>\nAnswer: 18\n"
        "Problem: 42 / 6\n<thought>6*7=42, so 42/6=7.</thought>\nAnswer: 7\n"
        "Problem: 96 / 8\n<thought>8*12=96, so 96/8=12.</thought>\nAnswer: 12\n"
        f"Problem: {problem_str}\n<thought>"
    )


def _first_number_from(generated: str) -> float | None:
    nums = re.findall(r"-?\d+\.?\d*", generated)
    if not nums:
        return None
    try:
        return float(nums[0])
    except ValueError:
        return None


_ANSWER_RE = re.compile(r"Answer\s*:\s*(-?\d+\.?\d*)")


def _extract_answer(generated: str) -> float | None:
    """Pick the answer of the FIRST completed step block.

    Stops at the next 'Problem:' boundary so few-shot continuations don't leak.
    Prefers an explicit 'Answer: X' marker; falls back to the last number
    inside the thought block.
    """
    chunk = generated.split("\nProblem:", 1)[0]
    match = _ANSWER_RE.search(chunk)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    nums = re.findall(r"-?\d+\.?\d*", chunk)
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


def _generate(prompt: str, *, do_sample: bool, temperature: float = 0.7) -> float | None:
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs.input_ids.shape[1]
    gen_kwargs: dict = dict(
        max_new_tokens=80,
        do_sample=do_sample,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.9
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
    new_tokens = outputs[0][input_len:]
    generated = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _extract_answer(generated)


def predict(problem_str: str) -> float | None:
    prompt = _build_prompt(problem_str)
    greedy = _generate(prompt, do_sample=False)
    if SELF_CONSISTENCY <= 0:
        return greedy

    votes: list[float] = []
    if greedy is not None:
        votes.append(greedy)
    for _ in range(SELF_CONSISTENCY):
        s = _generate(prompt, do_sample=True, temperature=0.7)
        if s is not None:
            votes.append(s)
    if not votes:
        return None
    counter = Counter(votes)
    # Tie-breaker: prefer greedy answer when present, else most common.
    most_common, _ = counter.most_common(1)[0]
    return most_common


def main() -> None:
    test_cases = build_test_cases(NUM_PROBLEMS, seed=0)

    if VERBOSE:
        print(f"\nLocal Evaluation ({NUM_PROBLEMS} problems):", flush=True)
        print("=" * 40, flush=True)

    score = 0
    failures: list[tuple[str, float, float | None]] = []

    for i, (prob, expected) in enumerate(test_cases, start=1):
        pred = predict(prob)
        is_correct = pred is not None and abs(pred - expected) < 1e-4
        if is_correct:
            score += 1
        else:
            failures.append((prob, expected, pred))
        if VERBOSE:
            status = "OK " if is_correct else "BAD"
            print(
                f"{status} [{i}/{NUM_PROBLEMS}] {prob} | expected {expected} | got {pred}",
                flush=True,
            )

    if VERBOSE:
        print("=" * 40, flush=True)
        print(f"FINAL SCORE: {score}/{NUM_PROBLEMS}", flush=True)
        if failures:
            print(f"\nFailures ({len(failures)}); showing up to 10:", flush=True)
            for prob, expected, pred in failures[:10]:
                print(f"  - {prob} | expected {expected} | got {pred}", flush=True)
    else:
        print(f"{score}/{NUM_PROBLEMS}", flush=True)


if __name__ == "__main__":
    main()
