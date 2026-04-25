import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./math_grpo_mps_final"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16 if DEVICE == "mps" else torch.float32,
).to(DEVICE)

def test_model(problem_str):
    prompt = (
        f"Solve the following math problem. Provide your reasoning inside <thought> tags, "
        f"and then output the final numeric answer.\n"
        f"Problem: {problem_str}\n"
        f"Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract number after "Answer:"
    answer_part = response.split("Answer:")[-1]
    nums = re.findall(r"-?\d+\.?\d*", answer_part)
    print(f"   Raw Response: {response}")
    print(f"   Answer Part: {answer_part}")
    return response, float(nums[-1]) if nums else None

test_cases = [
    ("5 + 7", 12.0),
    ("45 + 38", 83.0),
    ("6 * 8", 48.0),
    ("3 * (4 + 5)", 27.0),
    ("sqrt(49)", 7.0),
]

print("\n📊 Local Evaluation:")
print("="*40)
score = 0
for prob, expected in test_cases:
    resp, pred = test_model(prob)
    is_correct = pred is not None and abs(pred - expected) < 1e-4
    if is_correct: score += 1
    status = "✅" if is_correct else "❌"
    print(f"{status} Problem: {prob} | Expected: {expected} | Got: {pred}")

print("="*40)
print(f"FINAL SCORE: {score}/{len(test_cases)}")
