import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import json
import re

MODEL_PATH = "./math_grpo_mps_final"
SERVER_URL = "http://localhost:8000"

def get_problem():
    try:
        resp = requests.post(f"{SERVER_URL}/reset")
        resp.raise_for_status()
        
        resp = requests.post(f"{SERVER_URL}/step", json={
            "action": {
                "type": "call_tool",
                "tool_name": "get_problem",
                "arguments": {}
            }
        })
        resp.raise_for_status()
        return resp.json()["observation"]["result"]["data"]
    except Exception as e:
        print(f"Error connecting to server: {e}")
        return None

def main():
    print(f"Loading trained model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    problem = get_problem()
    if not problem:
        return

    print(f"\n--- Current Problem ---\n{problem}")
    
    prompt = f"<|im_start|>system\nYou are a math assistant. Solve the problem step-by-step and provide the final answer using the submit_answer tool.<|im_end|>\n<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n<thought>\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("\nGenerating solution...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print("\n--- Model Response ---\n")
    print(response)

    # Extract answer from response
    # Look for submit_answer({"answer": ...}) or just the answer
    match = re.search(r'submit_answer\(.*?"answer":\s*(\d+).*?\)', response)
    if match:
        answer = match.group(1)
        print(f"\nExtracted Answer: {answer}")
        
        # Submit to server
        resp = requests.post(f"{SERVER_URL}/step", json={
            "action": {
                "type": "call_tool",
                "tool_name": "submit_answer",
                "arguments": {"answer": int(answer)}
            }
        })
        print(f"Server Response: {resp.json()['observation']['result']['data']}")
        print(f"Reward: {resp.json()['observation']['reward']}")
    else:
        print("\nCould not find a structured answer submission.")

if __name__ == "__main__":
    main()
