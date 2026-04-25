# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Evolving Coding Competition — Theme #4: Self-Improvement

The agent is presented with increasingly complex coding challenges.
It submits Python code as a string; the environment runs the code
against hidden test cases in a sandboxed exec() environment.

Difficulty progression (1-10):
  1. Return a number
  2. Sum a list
  3. Find maximum in a list
  4. Filter even numbers
  5. Reverse a string
  6. Check if palindrome
  7. Binary search
  8. Count word frequencies
  9. Flatten nested list
  10. Find all prime numbers up to N

Reward design (multi-component):
  +1.0  all tests pass
  +0.5  bonus per test above 50% passing (partial credit)
  -0.3  syntax/runtime error
  -0.1  per failed test case
  +0.5  level-clear bonus (every 2 challenges solved fully)
  +0.01 record_thought (capped at 5)

Anti-hack:
  - exec() runs in an isolated namespace (no imports except builtins)
  - 2-second timeout simulation (step budget)
  - Forbidden: __import__, open, os, sys, subprocess
  - Code length capped at 1000 chars
"""

from typing import Any, Optional
from uuid import uuid4
import random
import time

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_CODE_STEPS = 100
CODE_MAX_CHARS = 1000

# Forbidden patterns in submitted code (anti-cheat)
FORBIDDEN = ["__import__", "open(", "os.", "sys.", "subprocess", "eval(", "exec(", "compile("]


# ---------------------------------------------------------------------------
# Challenge definitions — each has a description, test cases, and solution
# ---------------------------------------------------------------------------

def _get_challenges() -> dict:
    return {
        1: {
            "title": "Double the number",
            "description": "Write a function called `solve(n)` that returns n * 2.",
            "tests": [(2, 4), (5, 10), (0, 0), (-3, -6), (100, 200)],
            "hint": "Just return n * 2."
        },
        2: {
            "title": "Sum a list",
            "description": "Write `solve(lst)` that returns the sum of all numbers in lst.",
            "tests": [([1, 2, 3], 6), ([0], 0), ([-1, 1], 0), ([10, 20, 30], 60)],
            "hint": "Use the built-in sum() function."
        },
        3: {
            "title": "Find the maximum",
            "description": "Write `solve(lst)` that returns the largest number in lst.",
            "tests": [([3, 1, 4, 1, 5, 9], 9), ([7], 7), ([-1, -5, -2], -1), ([0, 0], 0)],
            "hint": "Use the built-in max() function."
        },
        4: {
            "title": "Filter even numbers",
            "description": "Write `solve(lst)` that returns a list of only even numbers from lst.",
            "tests": [([1,2,3,4,5,6], [2,4,6]), ([1,3,5], []), ([2,4], [2,4]), ([0], [0])],
            "hint": "Use a list comprehension with n % 2 == 0."
        },
        5: {
            "title": "Reverse a string",
            "description": "Write `solve(s)` that returns the string s reversed.",
            "tests": [("hello", "olleh"), ("abc", "cba"), ("", ""), ("a", "a"), ("racecar", "racecar")],
            "hint": "Use Python slicing: s[::-1]"
        },
        6: {
            "title": "Check palindrome",
            "description": "Write `solve(s)` that returns True if s is a palindrome, False otherwise. Ignore case.",
            "tests": [("racecar", True), ("hello", False), ("Madam", True), ("level", True), ("world", False)],
            "hint": "Compare s.lower() with s.lower()[::-1]"
        },
        7: {
            "title": "Binary search",
            "description": (
                "Write `solve(lst, target)` that returns the INDEX of target in the "
                "sorted list lst using binary search, or -1 if not found."
            ),
            "tests": [
                ([1, 3, 5, 7, 9], 5, 2),
                ([1, 3, 5, 7, 9], 1, 0),
                ([1, 3, 5, 7, 9], 9, 4),
                ([1, 3, 5, 7, 9], 4, -1),
            ],
            "hint": "Use low, high, mid pointers. Check mid value each iteration."
        },
        8: {
            "title": "Word frequency count",
            "description": (
                "Write `solve(text)` that returns a dict of word → count. "
                "Words are lowercase, split by spaces."
            ),
            "tests": [
                ("hello world hello", {"hello": 2, "world": 1}),
                ("a b c a", {"a": 2, "b": 1, "c": 1}),
                ("one", {"one": 1}),
            ],
            "hint": "Split by spaces, lowercase each word, use a dict to count."
        },
        9: {
            "title": "Flatten nested list",
            "description": "Write `solve(lst)` that flattens ONE level of nesting. [[1,2],[3,4]] → [1,2,3,4]",
            "tests": [
                ([[1,2],[3,4]], [1,2,3,4]),
                ([[1],[2,3],[4]], [1,2,3,4]),
                ([[5]], [5]),
                ([[], [1,2]], [1,2]),
            ],
            "hint": "Use a list comprehension: [item for sublist in lst for item in sublist]"
        },
        10: {
            "title": "Primes up to N",
            "description": "Write `solve(n)` that returns a sorted list of all prime numbers from 2 to n (inclusive).",
            "tests": [
                (10, [2, 3, 5, 7]),
                (20, [2, 3, 5, 7, 11, 13, 17, 19]),
                (2, [2]),
                (1, []),
                (30, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]),
            ],
            "hint": "For each number from 2 to n, check if any number 2..sqrt(i) divides it."
        },
    }

# ---------------------------------------------------------------------------
# Global Session Registry
# ---------------------------------------------------------------------------
_SESSIONS: dict[str, dict] = {}

def get_session(episode_id: str) -> dict:
    if not episode_id:
        episode_id = "default"
    if episode_id not in _SESSIONS:
        _SESSIONS[episode_id] = {
            "difficulty": 1,
            "challenges_solved": 0,
            "total_attempts": 0,
            "step_count": 0,
            "episode_id": episode_id,
            "last_reward": 0.0,
            "thoughts": [],
            "current_challenge": {},
        }
    return _SESSIONS[episode_id]

class CodingEnvironment(MCPEnvironment):
    """
    Evolving coding competition: agent submits Python code for increasingly
    complex algorithmic challenges.
    
    Tools:
      get_challenge()            — observe current coding problem
      submit_code(code)          — submit Python solution, runs all tests
      get_hint()                 — get a hint (-0.2 reward)
      record_thought(text)       — chain-of-thought reasoning (+0.01)
      get_status()               — current progress
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.mcp = FastMCP("coding_env")
        self._register_tools()

    def _register_tools(self):
        @self.mcp.tool()
        def get_challenge(episode_id: str = "default") -> str:
            s = get_session(episode_id)
            if not s["current_challenge"]: self._load_challenge(s)
            ch = s["current_challenge"]
            d = s["difficulty"]
            n_tests = len(ch.get("tests", []))
            return (
                f"[Difficulty {d}/10] {ch['title']}\n\n"
                f"Task: {ch['description']}\n\n"
                f"Your function will be tested against {n_tests} test cases.\n"
                f"Write a Python function named `solve` and submit with submit_code(code)."
            )

        @self.mcp.tool()
        def submit_code(code: str, episode_id: str = "default") -> str:
            s = get_session(episode_id)
            if not s["current_challenge"]: self._load_challenge(s)
            ch = s["current_challenge"]
            tests = ch.get("tests", [])
            s["total_attempts"] += 1

            for forbidden in FORBIDDEN:
                if forbidden in code:
                    s["last_reward"] = -0.5
                    return f"❌ REJECTED: Forbidden pattern '{forbidden}'."

            if len(code) > CODE_MAX_CHARS:
                s["last_reward"] = -0.3
                return "❌ Code too long."

            passed = 0
            errors = []
            for i, test in enumerate(tests):
                try:
                    safe_builtins = {
                        "range": range, "len": len, "int": int, "float": float,
                        "str": str, "list": list, "dict": dict, "set": set,
                        "tuple": tuple, "sorted": sorted, "sum": sum,
                        "min": min, "max": max, "abs": abs,
                        "enumerate": enumerate, "zip": zip, "map": map,
                        "filter": filter, "isinstance": isinstance,
                        "bool": bool, "print": print, "True": True, "False": False, "None": None,
                        "math": __import__("math")
                    }
                    ns = {"__builtins__": safe_builtins}
                    exec(compile(code, "<code>", "exec"), ns)
                    
                    if len(test) == 2:
                        inputs, expected = test
                        actual = ns["solve"](inputs)
                    else:
                        inp1, inp2, expected = test
                        actual = ns["solve"](inp1, inp2)
                        
                    if actual == expected:
                        passed += 1
                    else:
                        errors.append(f"Test {i+1} failed: expected {expected!r}, got {actual!r}")
                except Exception as e:
                    errors.append(f"Test {i+1} error: {e}")

            total = len(tests)
            if passed == total:
                s["challenges_solved"] += 1
                reward = 1.5
                if s["challenges_solved"] % 2 == 0:
                    s["difficulty"] = min(s["difficulty"] + 1, 10)
                    reward += 0.5
                self._load_challenge(s)
                s["last_reward"] = reward
                return f"✅ ALL {total}/{total} PASSED! Reward: +{reward:.2f}."
            else:
                pass_rate = passed / total
                s["last_reward"] = (pass_rate * 0.5) - 0.2
                return f"⚠️ {passed}/{total} passed. Reward: {s['last_reward']:+.2f}."

        @self.mcp.tool()
        def get_hint(episode_id: str = "default") -> str:
            s = get_session(episode_id)
            if not s["current_challenge"]: self._load_challenge(s)
            s["last_reward"] = -0.2
            return f"Hint: {s['current_challenge'].get('hint', 'Think step by step.')}"

        @self.mcp.tool()
        def record_thought(thought: str, episode_id: str = "default") -> str:
            s = get_session(episode_id)
            if len(s["thoughts"]) < 5: s["last_reward"] = 0.01
            s["thoughts"].append(thought[:500])
            return f"Thought recorded ({len(s['thoughts'])}/5)."

        @self.mcp.tool()
        def get_status(episode_id: str = "default") -> str:
            s = get_session(episode_id)
            return f"Diff: {s['difficulty']} | Solved: {s['challenges_solved']} | Steps: {s['step_count']}"

    def _load_challenge(self, s: dict) -> None:
        d = s["difficulty"]
        challenges = _get_challenges()
        s["current_challenge"] = challenges.get(min(d, 10), challenges[10])

    def reset(self, episode_id: Optional[str] = None, difficulty: Optional[int] = None, **kwargs) -> Observation:
        eid = episode_id or str(uuid4())
        s = get_session(eid)
        if difficulty is not None:
            s["difficulty"] = difficulty
        else:
            s["difficulty"] = 1
            s["challenges_solved"] = 0
        s["step_count"] = 0
        self._load_challenge(s)
        return Observation(
            done=False, reward=0.0,
            metadata={"episode_id": eid, "status": "ready", "difficulty": s["difficulty"]}
        )

    def _step_impl(self, action: Action, **kwargs) -> Observation:
        raise NotImplementedError("Only MCP actions are supported")

    def step(self, action: Action, **kwargs) -> Observation:
        eid = kwargs.get("episode_id", "default")
        s = get_session(eid)
        obs = super().step(action, **kwargs)
        s["step_count"] += 1
        return Observation(
            done=s["difficulty"] > 10 or s["step_count"] > 100,
            reward=s["last_reward"],
            metadata={
                "observation": getattr(obs, "result", None) or getattr(obs, "observation", None),
                "difficulty": s["difficulty"],
                "episode_id": eid,
                "step": s["step_count"]
            }
        )

    @property
    def state(self) -> State:
        return State()
