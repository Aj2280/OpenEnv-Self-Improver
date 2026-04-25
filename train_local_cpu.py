import os
import torch
import requests
import re
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import datasets

BASE_URL = "http://localhost:8000"

# 1. Load a tiny model for local CPU verification
MODEL_NAME = "openai-community/gpt2" # Very small, fits on CPU

print(f"Loading tokenizer and model: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# 2. Reward function (Same as in TRAINING.md)
def call_env(tool, args={}):
    try:
        r = requests.post(f"{BASE_URL}/step", json={
            "action": {"type": "call_tool", "tool_name": tool, "arguments": args}
        }, timeout=5)
        return r.json()
    except Exception as e:
        print(f"Error calling environment: {e}")
        return {"reward": -0.5}

def compute_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        # TRL passes completions in various formats; handle list of content
        if isinstance(completion, list):
            text = completion[0]["content"]
        else:
            text = str(completion)
            
        nums = re.findall(r"-?\d+\.?\d*", text)
        if not nums:
            rewards.append(-0.5)
            continue
            
        try:
            answer = float(nums[-1])
            resp = call_env("submit_answer", {"answer": answer})
            reward = resp.get("reward", -0.5)
            rewards.append(float(reward))
        except Exception:
            rewards.append(-0.5)
    return rewards

# 3. Dataset building (Small sample for local run)
def build_dataset(n_problems=10):
    data = []
    print(f"Building dataset of {n_problems} problems...")
    requests.post(f"{BASE_URL}/reset", json={})
    
    for i in range(n_problems):
        resp = call_env("get_problem")
        obs = resp.get("observation", {})
        problem_str = obs.get("result", {}).get("data", "")
        
        prompt = (
            f"Solve: {problem_str}\n"
            f"Answer:"
        )
        data.append({"prompt": prompt})
        call_env("submit_answer", {"answer": -999.0}) # Move to next
    
    return datasets.Dataset.from_list(data)

# 4. Training configuration (Minimal for local CPU)
training_args = GRPOConfig(
    output_dir="./local_test_grpo",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    max_steps=5, # Just run a few steps to verify
    logging_steps=1,
    report_to="none",
    num_generations=2,
    max_new_tokens=20,
    temperature=0.7,
)

# 5. Execute training
if __name__ == "__main__":
    print("🚀 Starting local CPU 'Mock Training' to verify loop...")
    dataset = build_dataset(5)
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=compute_reward,
        processing_class=tokenizer,
    )
    
    print("Training loop active. Running verification steps...")
    trainer.train()
    print("✅ Local verification complete! The TRL/GRPO loop is functional.")
    print("Now proceed to Google Colab for the real training with Unsloth.")
