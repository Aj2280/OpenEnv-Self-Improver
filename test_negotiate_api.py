import requests
BASE_URL = "http://localhost:8000/negotiate"
print("RESET:", requests.post(f"{BASE_URL}/reset", json={}).json())
print("STATE:", requests.post(f"{BASE_URL}/step", json={"action": {"type": "call_tool", "tool_name": "get_negotiation_state"}}).json())
print("OFFER:", requests.post(f"{BASE_URL}/step", json={"action": {"type": "call_tool", "tool_name": "make_offer", "arguments": {"resource": "food", "amount": 60}}}).json())
print("FINALIZE:", requests.post(f"{BASE_URL}/step", json={"action": {"type": "call_tool", "tool_name": "finalize_offer"}}).json())
