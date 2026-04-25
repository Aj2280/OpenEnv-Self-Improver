import os
import torch
import requests
import re
import math
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import datasets

BASE_URL = "http://localhost:8000/code"

# 1. Model Selection - Small enough for Mac MPS
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 

print(f"🚀 Initializing Coding Competition Training on MPS...")
print(f"📦 Model: {MODEL_NAME}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"🖥️  Using device: {device}")

# 2. Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "mps" else torch.float32,
).to(device)

# 3. Environment Interaction
def call_env(tool, args={}, episode_id=None):
    try:
        payload = {
            "action": {"type": "call_tool", "tool_name": tool, "arguments": args}
        }
        if episode_id:
            payload["episode_id"] = episode_id
            
        r = requests.post(f"{BASE_URL}/step", json=payload, timeout=10)
        return r.json()
    except Exception as e:
        print(f"Error calling env: {e}")
        return {"reward": -0.5}

def compute_reward(completions, prompts, **kwargs):
    """
    Reward function for coding tasks.
    Uses unique episode IDs per rollout to prevent session interference.
    """
    import uuid
    rewards = []
    for completion, prompt in zip(completions, prompts):
        # Extract text from completion
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            
        # Extract code from markdown block
        code_match = re.search(r"```python\n(.*?)\n```", text, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            if "def solve" in text:
                code = text
            else:
                rewards.append(-0.5)
                continue
            
        try:
            # Extract difficulty from prompt to sync server state
            diff_match = re.search(r"\[Difficulty (\d+)/10\]", prompt)
            difficulty = int(diff_match.group(1)) if diff_match else 1
            
            eid = f"code_rollout_{uuid.uuid4()}"
            requests.post(f"http://localhost:8000/code/reset", json={"episode_id": eid, "difficulty": difficulty})
            
            resp = call_env("submit_code", {"code": code}, episode_id=eid)
            # The server returns Observation object in response to step
            reward = resp.get("reward", -0.5)
            rewards.append(float(reward))
        except Exception:
            rewards.append(-0.5)
    
    return rewards

# 4. Curriculum Dataset Building
def build_dataset(n_problems=40):
    print(f"📊 Sampling {n_problems} challenges from curriculum...")
    data = []
    requests.post(f"{BASE_URL}/reset", json={})
    
    for i in range(n_problems):
        resp = call_env("get_challenge")
        obs = resp.get("observation", {})
        challenge_str = obs.get("text", "")
        
        prompt = (
            f"Solve the following Python coding challenge.\n"
            f"You must define a function named `solve` that passes the test cases.\n"
            f"Output your code inside a ```python markdown block.\n\n"
            f"Challenge:\n{challenge_str}\n\n"
            f"Solution:"
        )
        data.append({"prompt": prompt})
        
        # Advance to next challenge (submit dummy to fail/pass)
        # Note: we need to actually solve or skip.
        # For building the dataset, we just want the prompt.
        # But if we want to sample from different levels, we need to solve them.
        # Since we're training, we'll just sample level 1 mostly.
        if (i+1) % 10 == 0:
            print(f"  Collected {i+1}/{n_problems}")
    
    return datasets.Dataset.from_list(data)

# 5. GRPO Configuration
training_args = GRPOConfig(
    output_dir="./code_grpo_mps",
    num_train_epochs=1,
    per_device_train_batch_size=1, # Coding takes more memory
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    max_steps=30, # Start small
    logging_steps=5,
    save_steps=30,
    report_to="none",
    # GRPO Specifics
    num_generations=4,
    max_completion_length=256, # Coding needs more length
    temperature=0.7,
)

# 6. Run Training
if __name__ == "__main__":
    # Ensure server is up
    try:
        requests.get(f"http://localhost:8000/health", timeout=2)
    except:
        print(f"❌ Server not found at http://localhost:8000. Start it with 'uv run --project . server' first.")
        exit(1)

    dataset = build_dataset(40)
    
    print("✨ Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=compute_reward,
        processing_class=tokenizer,
    )
    
    print("🔥 Starting training loop...")
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    print(f"✅ Training complete in {(end_time - start_time)/60:.1f} minutes!")
    trainer.save_model("./code_grpo_mps_final")
    print("💾 Model saved to ./code_grpo_mps_final")
