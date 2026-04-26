# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import random
import time
import re
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Observation, Action, State
from openenv.core.env_server.mcp_types import CallToolObservation
from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Global Session Registry (Guideline #12: Concurrency & Parallel Rollouts)
# ---------------------------------------------------------------------------
_SESSIONS: dict[str, dict] = {}

def get_session(episode_id: str) -> dict:
    """Retrieve or initialize a session state by episode_id."""
    if not episode_id:
        episode_id = "default"
    if episode_id not in _SESSIONS:
        _SESSIONS[episode_id] = {
            "difficulty": 1,
            "problems_solved": 0,
            "wrong_attempts": 0,
            "hints_used": 0,
            "step_count": 0,
            "current_problem": "",
            "expected_answer": 0.0,
            "thoughts": [],
            "last_reward": 0.0,
            "episode_id": episode_id,
            "episode_start": time.time(),
        }
        _generate_problem_for_session(_SESSIONS[episode_id])
    return _SESSIONS[episode_id]

def _generate_problem_for_session(s: dict) -> None:
    """Generate a fresh math problem for the session's current tier."""
    d = s["difficulty"]
    if d == 1:
        a, b = random.randint(1, 10), random.randint(1, 10)
        prob, ans = f"{a} + {b}", float(a + b)
    elif d == 2:
        a, b = random.randint(10, 99), random.randint(10, 99)
        prob, ans = f"{a} + {b}", float(a + b)
    elif d == 3:
        a, b = sorted([random.randint(10, 99), random.randint(10, 99)], reverse=True)
        prob, ans = f"{a} - {b}", float(a - b)
    elif d == 4:
        a, b = random.randint(2, 12), random.randint(2, 12)
        prob, ans = f"{a} * {b}", float(a * b)
    elif d == 5:
        a, b, c = random.randint(2, 10), random.randint(2, 10), random.randint(1, 20)
        prob, ans = f"{a} * {b} + {c}", float(a * b + c)
    elif d == 6:
        a, b, c = random.randint(2, 10), random.randint(2, 10), random.randint(1, 20)
        prob, ans = f"{a} * ({b} + {c})", float(a * (b + c))
    elif d == 7:
        a, b = random.randint(5, 20), random.randint(2, 9)
        prob, ans = f"{a * b} / {b}", float(a)
    elif d == 8:
        a, x = random.randint(2, 9), random.randint(1, 15)
        b = random.randint(1, 30)
        c = a * x + b
        prob = f"Solve for x: {a}x + {b} = {c}"
        ans = float(x)
    elif d == 9:
        x = random.randint(2, 20)
        prob, ans = f"sqrt({x * x})", float(x)
    else:
        a, x_val, div = random.randint(2, 5), random.randint(1, 10), random.randint(2, 4)
        b = div * random.randint(2, 8) - a * x_val
        for _ in range(10):
            if abs(b) <= 40: break
            b = div * random.randint(2, 8) - a * x_val
        e = (a * x_val + b) // div
        prob = f"Solve for x: ({a}x + {b}) / {div} = {e}"
        ans = float(x_val)

    s["current_problem"] = prob
    s["expected_answer"] = ans

# Reward Constants
R_CORRECT       =  1.0
R_FORMAT        =  0.1
R_LEVEL_CLEAR   =  0.5
R_WRONG         = -0.5
R_HINT          = -0.2
R_THOUGHT       =  0.01

class MathEscalationEnvironment(MCPEnvironment):
    """Adaptive 10-tier math curriculum with parallel session support."""
    
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self.mcp = FastMCP("math_escalation")
        self._register_tools()
        super().__init__(self.mcp)

    def _register_tools(self):
        @self.mcp.tool()
        def get_problem(episode_id: str = "default") -> str:
            s = get_session(episode_id)
            return f"[Difficulty {s['difficulty']}/10] Solve: {s['current_problem']}"

        @self.mcp.tool()
        def submit_answer(answer: float, episode_id: str = "default") -> str:
            s = get_session(episode_id)
            s["step_count"] += 1
            reward = R_FORMAT
            
            if abs(answer - s["expected_answer"]) < 1e-6:
                s["problems_solved"] += 1
                reward += R_CORRECT
                if s["problems_solved"] % 2 == 0:
                    s["difficulty"] = min(s["difficulty"] + 1, 10)
                    reward += R_LEVEL_CLEAR
                    _generate_problem_for_session(s)
                    msg = f"Correct! Level Up! Now Tier {s['difficulty']}."
                else:
                    _generate_problem_for_session(s)
                    msg = "Correct! Next problem ready."
            else:
                s["wrong_attempts"] += 1
                reward += R_WRONG
                msg = f"Incorrect. The expected answer was {s['expected_answer']}."
            
            s["last_reward"] = reward
            return msg

        @self.mcp.tool()
        def record_thought(thought: str, episode_id: str = "default") -> str:
            s = get_session(episode_id)
            s["thoughts"].append(thought)
            s["last_reward"] = R_THOUGHT if len(s["thoughts"]) < 50 else 0.0
            return "Thought recorded."

        @self.mcp.tool()
        def get_status(episode_id: str = "default") -> dict:
            s = get_session(episode_id)
            return {
                "tier": s["difficulty"],
                "solved": s["problems_solved"],
                "steps": s["step_count"]
            }

    def reset(self, episode_id: Optional[str] = None, difficulty: Optional[int] = None, problem: Optional[str] = None, **kwargs) -> Observation:
        eid = episode_id or str(uuid4())
        # Instead of deleting, we initialize/update the session
        s = get_session(eid)
        if difficulty is not None:
            s["difficulty"] = difficulty
            _generate_problem_for_session(s)
        if problem is not None:
            s["current_problem"] = problem
            # Note: we don't know the expected answer if we just set the problem string
            # So we might need a way to parse it, or just use the existing oracle
            # For now, let's assume if problem is set, we use an oracle to find the answer
            from .oracle import agent_solve # I'll create this file
            s["expected_answer"] = agent_solve(problem)
        else:
            # Traditional reset: fresh problem at current or start difficulty
            s["problems_solved"] = 0
            s["step_count"] = 0
            _generate_problem_for_session(s)

        return Observation(
            done=False, reward=0.0,
            metadata={"episode_id": eid, "status": "ready", "problem": s["current_problem"], "difficulty": s["difficulty"]}
        )

    @property
    def state(self) -> State:
        return State()

    def _step_impl(self, action: Action, **kwargs) -> Observation:
        raise NotImplementedError("Only MCP actions are supported")

    def step(self, action: Action, **kwargs) -> Observation:
        eid = kwargs.get("episode_id", "default")
        s = get_session(eid)
        
        # Execute tool via FastMCP
        obs = super().step(action, **kwargs)
        
        # Preserve the tool result fields, but attach environment reward/done/metadata.
        # `create_app(..., CallToolObservation)` expects tool_name/result/error at the top level.
        if not isinstance(obs, CallToolObservation):
            obs = CallToolObservation(tool_name="unknown", result=getattr(obs, "result", None), error=None)

        merged_metadata = dict(getattr(obs, "metadata", {}) or {})
        merged_metadata.update(
            {
                "difficulty": s["difficulty"],
                "episode_id": eid,
                "step": s["step_count"],
            }
        )

        return CallToolObservation(
            tool_name=obs.tool_name,
            result=obs.result,
            error=obs.error,
            done=s["step_count"] >= 200,
            reward=s["last_reward"],
            metadata=merged_metadata,
        )
