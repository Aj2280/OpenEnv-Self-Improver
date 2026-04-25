# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Math Escalation Environment Client.
"""

from openenv.core.mcp_client import MCPToolClient
import asyncio
import sys

class MathEnv(MCPToolClient):
    """
    Client for the Math Escalation Environment.
    """
    pass

async def interactive_loop():
    print("Connecting to Math Escalation Environment...")
    try:
        async with MathEnv(base_url="http://localhost:8000") as env:
            await env.reset()
            print("Connected! Type 'exit' to quit.\n")
            
            while True:
                problem = await env.call_tool("get_problem")
                print(f"\033[94m{problem}\033[0m")
                
                user_input = input("Your answer: ").strip()
                if user_input.lower() in ["exit", "quit"]:
                    break
                
                try:
                    answer = float(user_input)
                except ValueError:
                    print("\033[91mInvalid input. Please enter a number.\033[0m")
                    continue
                
                obs = await env.step({
                    "type": "call_tool",
                    "tool_name": "submit_answer",
                    "arguments": {"answer": answer}
                })
                
                print(f"Result: {obs.result}")
                print(f"Reward: {obs.reward} | Done: {obs.done}\n")
                
                if obs.done:
                    print("Congratulations! You've mastered all levels!")
                    break
    except Exception as e:
        print(f"\033[91mError: {e}\033[0m")
        print("Make sure the server is running with: uv run --project . server")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        asyncio.run(interactive_loop())
    else:
        # Default behavior for non-interactive scripts
        pass
