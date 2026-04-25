# Math Escalation Environment — TRL/GRPO Training Guide

This document explains how to connect **Hugging Face TRL + Unsloth** to the
Math Escalation Environment for real LLM fine-tuning with reinforcement learning.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  TRL GRPOTrainer  (optimizer, policy update)        │
│  ┌───────────────────────────────────────────────┐  │
│  │  Unsloth  (fast inference + memory savings)   │  │
│  └────────────────────┬──────────────────────────┘  │
│                       │ generate(prompt)             │
└───────────────────────┼─────────────────────────────┘
                        │
     ┌──────────────────▼──────────────────┐
     │  Math Escalation Environment        │
     │  (OpenEnv / FastAPI / MCP)          │
     │  http://localhost:8000              │
     │                                     │
     │  Tools: get_problem()               │
     │         submit_answer(answer)       │
     │         record_thought(text)        │
     │         get_hint()                  │
     │         get_status()               │
     └─────────────────────────────────────┘
```

---

## Quick Start (Google Colab)

```python
# Cell 1 — Install dependencies
!pip install "unsloth[colab-new]" trl>=0.8.0 requests

# Cell 2 — Start the environment server (run locally or on HF Spaces)
# Locally: uv run --project . server
# HF Spaces: automatically started by Dockerfile

BASE_URL = "http://localhost:8000"  # change to HF Space URL

# Cell 3 — Load model with Unsloth
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-1.5B-Instruct",
    max_seq_length = 1024,
    dtype = None,
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "v_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
)

# Cell 4 — Define the environment reward function
import requests, re, math

def call_env(tool, args={}):
    r = requests.post(f"{BASE_URL}/step", json={
        "action": {"type": "call_tool", "tool_name": tool, "arguments": args}
    })
    return r.json()

def compute_reward(completions, **kwargs):
    """
    Multi-component reward function (Guideline #7).
    Called by GRPOTrainer for each generated completion.
    """
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else completion
        
        # Try to extract a number from the model's output
        nums = re.findall(r"-?\d+\.?\d*", text)
        if not nums:
            rewards.append(-0.5)  # format penalty
            continue
            
        try:
            answer = float(nums[-1])
            resp = call_env("submit_answer", {"answer": answer})
            reward = resp.get("reward", -0.5)
            rewards.append(float(reward))
        except Exception:
            rewards.append(-0.5)
    
    return rewards

# Cell 5 — Build dataset from environment
def build_dataset(n_problems=200):
    """Sample problems from the environment at different difficulty levels."""
    import datasets
    
    data = []
    requests.post(f"{BASE_URL}/reset", json={})
    
    for i in range(n_problems):
        resp = call_env("get_problem")
        obs = resp.get("observation", {})
        problem_str = obs.get("result", {}).get("data", "")
        difficulty = obs.get("metadata", {}).get("difficulty", 1)
        
        prompt = (
            f"You are a math expert. Solve this problem and output ONLY the number.\n"
            f"{problem_str}\n"
            f"Answer:"
        )
        data.append({
            "prompt": prompt,
            "difficulty": difficulty,
        })
        
        # Move on (submit wrong answer to get next problem)
        call_env("submit_answer", {"answer": -999.0})
    
    return datasets.Dataset.from_list(data)

dataset = build_dataset(200)
print(f"Dataset: {len(dataset)} problems across difficulty tiers")

# Cell 6 — GRPO Training
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    output_dir = "./math_escalation_grpo",
    num_train_epochs = 3,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate = 5e-5,
    logging_steps = 10,
    save_steps = 100,
    warmup_ratio = 0.05,
    report_to = "none",
    # GRPO-specific
    num_generations = 4,          # rollouts per prompt
    max_new_tokens = 64,
    temperature = 0.7,
)

trainer = GRPOTrainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    reward_funcs = compute_reward,
    processing_class = tokenizer,
)

trainer.train()
trainer.save_model("./math_escalation_grpo_final")
print("Training complete!")
```

---

## Reward Components

| Signal | Value | Trigger |
|--------|-------|---------|
| Correct answer | +1.0 | Exact match |
| Format OK | +0.1 | Valid float submitted |
| Level clear bonus | +0.5 | Every 2nd correct answer |
| Wrong answer | -0.5 | Numeric mismatch |
| Hint used | -0.2 | `get_hint()` called |
| Thought recorded | +0.01 | `record_thought()` (capped at 5) |

---

## Expected Reward Curve Shape

A successfully training agent shows:
1. **Flat or negative** rewards early (tier 1-3 random attempts)
2. **Positive spike** as the model learns arithmetic format
3. **Plateau** at each tier boundary as it hits harder problems
4. **Step-wise increase** as it masters each tier

This "staircase" pattern is the signature of successful curriculum learning.
## Next Steps

- Run the provided Colab notebook `train_llm.ipynb` to perform full RL training on a larger model.
- Monitor the reward curve saved at `plots/reward_curve.png`.
- After training, evaluate the model on a held‑out set of problems using `evaluate.py`.
- Deploy the fine‑tuned model with OpenAI‑compatible API or as a FastAPI endpoint.

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Server not reachable | Wrong `BASE_URL` | Ensure the FastAPI server is running (`uv run server`) and the port matches. |
| No reward improvement | Reward function too sparse | Add additional components (e.g., time penalty, hint penalty) in `compute_reward`. |
| Out‑of‑memory | Model size too large for GPU | Use `load_in_4bit=True` or switch to a smaller base model. |

For further details see the README and the OpenEnv documentation.

---

*Happy training!*
