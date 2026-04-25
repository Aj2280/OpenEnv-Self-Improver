import requests, re, math
import datasets

BASE_URL = "http://localhost:8000"

def call_env(tool, args={}):
    r = requests.post(f"{BASE_URL}/step", json={
        "action": {"type": "call_tool", "tool_name": tool, "arguments": args}
    })
    return r.json()

def build_dataset(n_problems=50):
    """Sample problems from the environment at different difficulty levels."""
    data = []
    print(f"Resetting environment at {BASE_URL}...")
    requests.post(f"{BASE_URL}/reset", json={})
    
    print(f"Sampling {n_problems} problems...")
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
        if (i+1) % 10 == 0:
            print(f"  Collected {i+1}/{n_problems}")
    
    return datasets.Dataset.from_list(data)

if __name__ == "__main__":
    try:
        ds = build_dataset(20)
        print(f"✅ Successfully built dataset with {len(ds)} problems.")
        print(f"Sample prompt: {ds[0]['prompt']}")
    except Exception as e:
        print(f"❌ Failed to build dataset: {e}")
