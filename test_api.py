import requests
BASE_URL = "http://localhost:8000"
requests.post(f"{BASE_URL}/reset")
print("GET_PROBLEM:", requests.post(f"{BASE_URL}/step", json={"action": {"type": "call_tool", "tool_name": "get_problem"}}).json())
print("SUBMIT:", requests.post(f"{BASE_URL}/step", json={"action": {"type": "call_tool", "tool_name": "submit_answer", "arguments": {"answer": 0.0}}}).json())
