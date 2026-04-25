import os
import torch
import requests
import re
import math
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import datasets

BASE_URL = "http://localhost:8000"

# 1. Model Selection - Small enough for Mac MPS
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 

print(f"🚀 Initializing Math Escalation Training on MPS...")
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

# 3. Environment Interaction (Theme #4: Self-Improvement)
def call_env(tool, args={}, episode_id=None):
    try:
        payload = {
            "action": {"type": "call_tool", "tool_name": tool, "arguments": args}
        }
        if episode_id:
            payload["episode_id"] = episode_id
            
        r = requests.post(f"{BASE_URL}/step", json=payload, timeout=5)
        return r.json()
    except Exception as e:
        return {"reward": -0.5}

def compute_reward(completions, prompts, **kwargs):
    """
    Multi-component reward function.
    Uses unique episode IDs per rollout to prevent session interference.
    """
    import uuid
    rewards = []
    for completion, prompt in zip(completions, prompts):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        
        # Component 1: Formatting Reward (Reasoning Tags)
        # Signficantly boosted to prevent early collapse
        format_reward = 0.0
        if "<thought>" in text and "</thought>" in text:
            format_reward += 0.5
            # Bonus for non-empty thought
            thought_match = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL)
            if thought_match and len(thought_match.group(1).strip()) > 10:
                format_reward += 0.2
        
        # Component 2: Numeric Presence Reward
        nums = re.findall(r"-?\d+\.?\d*", text)
        numeric_reward = 0.3 if nums else 0.0
        
        # Component 3: Environment Correctness
        env_reward = 0.0
        if nums:
            try:
                prob_match = re.search(r"Problem: (.*?)\n", prompt)
                problem_str = prob_match.group(1) if prob_match else ""
                
                eid = f"mps_rollout_{uuid.uuid4()}"
                requests.post(f"{BASE_URL}/reset", json={"episode_id": eid, "problem": problem_str})
                
                answer = float(nums[-1])
                resp = call_env("submit_answer", {"answer": answer}, episode_id=eid)
                env_reward = float(resp.get("reward", -0.5))
            except Exception:
                env_reward = -0.5
        else:
            env_reward = -0.5

        # Total reward range: [-0.5, 2.0]
        total = env_reward + format_reward + numeric_reward
        rewards.append(total)
    
    return rewards

# 4. Curriculum Dataset Building
def build_dataset(n_problems=100):
    print(f"📊 Sampling {n_problems} problems from curriculum...")
    data = []
    requests.post(f"{BASE_URL}/reset", json={})
    
    for i in range(n_problems):
        resp = call_env("get_problem")
        obs = resp.get("observation", {})
        problem_str = obs.get("result", {}).get("data", "")
        
        prompt = (
            f"Solve the following math problem. Provide your reasoning inside <thought> tags, "
            f"and then output the final numeric answer.\n"
            f"Problem: {problem_str}\n"
            f"Answer:"
        )
        data.append({"prompt": prompt})
        
        # Advance to next problem
        call_env("submit_answer", {"answer": -9999.0})
        if (i+1) % 25 == 0:
            print(f"  Collected {i+1}/{n_problems}")
    
    return datasets.Dataset.from_list(data)

# 5. GRPO Configuration
training_args = GRPOConfig(
    output_dir="./math_grpo_mps",
    num_train_epochs=1,
    per_device_train_batch_size=1, # Reduce for larger num_generations
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    max_steps=100, 
    logging_steps=1,
    save_steps=100,
    report_to="none",
    # GRPO Specifics
    num_generations=8, # Increased for better relative reward signal
    max_completion_length=128, # Increased for reasoning
    temperature=0.9,
)

# 6. Run Training
if __name__ == "__main__":
    # Ensure server is up
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except:
        print(f"❌ Server not found at {BASE_URL}. Start it with 'uv run --project . server' first.")
        exit(1)

    dataset = build_dataset(100)
    
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
    trainer.save_model("./math_grpo_mps_final")
    print("💾 Model saved to ./math_grpo_mps_final")
