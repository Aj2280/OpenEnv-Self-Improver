import os
import torch
import requests
import re
import math
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import datasets

BASE_URL = "http://localhost:8000/negotiate"

# 1. Model Selection
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 

print(f"🚀 Initializing Negotiation Arena Training on MPS...")
print(f"📦 Model: {MODEL_NAME}")

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
).to(device)

# 3. Environment Interaction
def call_env(tool, args={}):
    try:
        r = requests.post(f"{BASE_URL}/step", json={
            "action": {"type": "call_tool", "tool_name": tool, "arguments": args}
        }, timeout=5)
        return r.json()
    except Exception:
        return {"reward": -0.5}

def compute_reward(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        
        # Look for "Offer: item=val, item=val..."
        # Example: "Offer: food=60, water=40"
        try:
            # Parse resource offers
            matches = re.findall(r"(\w+)=(\d+)", text)
            if not matches:
                # Check if it accepted
                if "accept" in text.lower():
                    resp = call_env("accept_opponent_offer")
                    rewards.append(float(resp.get("observation", {}).get("reward", -0.5)))
                else:
                    rewards.append(-0.5)
                continue
            
            # Make offers
            for res, amt in matches:
                call_env("make_offer", {"resource": res, "amount": int(amt)})
            
            # Finalize
            resp = call_env("finalize_offer")
            reward = resp.get("observation", {}).get("reward", -0.5)
            rewards.append(float(reward))
        except Exception:
            rewards.append(-0.5)
    
    return rewards

# 4. Dataset Building
def build_dataset(n_scenarios=40):
    print(f"📊 Sampling {n_scenarios} negotiation scenarios...")
    data = []
    requests.post(f"{BASE_URL}/reset", json={})
    
    for i in range(n_scenarios):
        resp = call_env("get_negotiation_state")
        obs = resp.get("observation", {})
        state_str = obs.get("text", "")
        
        prompt = (
            f"You are in a negotiation. Here is the state:\n{state_str}\n\n"
            f"Decide your share of the resources. "
            f"Output your offer in the format 'Offer: resource=amount, ...' "
            f"or say 'accept' to accept the opponent's offer.\n"
            f"Your goal is to maximize your total share while ensuring the opponent accepts.\n"
            f"Response:"
        )
        data.append({"prompt": prompt})
        
        # Reset/Next scenario
        call_env("accept_opponent_offer")
        if (i+1) % 10 == 0:
            print(f"  Collected {i+1}/{n_scenarios}")
    
    return datasets.Dataset.from_list(data)

# 5. GRPO Configuration
training_args = GRPOConfig(
    output_dir="./neg_grpo_mps",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    max_steps=30,
    logging_steps=5,
    save_steps=30,
    report_to="none",
    num_generations=4,
    max_completion_length=128,
    temperature=0.7,
)

if __name__ == "__main__":
    try:
        requests.get("http://localhost:8000/health", timeout=2)
    except:
        print("❌ Server not found.")
        exit(1)

    dataset = build_dataset(40)
    
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=compute_reward,
        processing_class=tokenizer,
    )
    
    print("🔥 Starting negotiation training...")
    trainer.train()
    trainer.save_model("./neg_grpo_mps_final")
    print("💾 Model saved to ./neg_grpo_mps_final")
