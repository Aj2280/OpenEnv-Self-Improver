import requests
BASE_URL = "http://localhost:8000/code"
requests.post(f"{BASE_URL}/reset")
print("GET_CHALLENGE:", requests.post(f"{BASE_URL}/step", json={"action": {"type": "call_tool", "tool_name": "get_challenge"}}).json())
