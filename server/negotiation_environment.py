# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Self-Play Negotiation Arena — Theme #4: Self-Improvement

Two agents negotiate over a bundle of resources. The agent plays as
the "Proposer"; a built-in rule-based opponent plays as "Responder."

As difficulty escalates (1-10):
  - More resource types are added (1 → 5 items)
  - Opponent becomes greedier (lower acceptance threshold)
  - Fewer rounds are allowed to reach a deal

Self-improvement signal: the agent learns to make strategic offers
that the opponent will accept while maximizing its own share.

Reward design (multi-component):
  +1.5  deal reached
  +share_quality  agent's normalized share of total value
  -0.3  per failed offer (encourages efficiency)
  -1.0  no deal reached (episode ends without agreement)
  +0.1  record_thought (capped at 3)
"""

from typing import Any, Optional
from uuid import uuid4
import random
import time

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action, Observation, State
from fastmcp import FastMCP

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
            "round": 0,
            "max_rounds": MAX_ROUNDS,
            "deals_made": 0,
            "failed_deals": 0,
            "episode_id": episode_id,
            "step_count": 0,
            "last_reward": 0.0,
            "thoughts": [],
            "deal_reached": False,
            "current_resources": {},
            "agent_share": {},
            "opponent_offer": {},
            "history": [],
        }
    return _SESSIONS[episode_id]

class NegotiationEnvironment(MCPEnvironment):
    """
    Self-play negotiation arena: agent vs. rule-based opponent.
    
    Tools:
      get_negotiation_state()        — see current offer + round info
      make_offer(resource, amount)   — propose your share of one resource
      finalize_offer()               — submit current proposal to opponent
      accept_opponent_offer()        — accept the opponent's current offer
      record_thought(text)           — chain-of-thought (small reward)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self.mcp = FastMCP("negotiation_env")
        self._register_tools()
        super().__init__(self.mcp)

    def _register_tools(self):
        @self.mcp.tool()
        def get_negotiation_state(episode_id: str = "default") -> str:
            s = get_session(episode_id)
            if not s["current_resources"]:
                self._generate_negotiation(s)
            
            resources = s["current_resources"]
            opp = s["opponent_offer"]
            draft = s["agent_share"]
            remaining = s["max_rounds"] - s["round"]
            d = s["difficulty"]

            lines = [
                f"[Difficulty {d}/10] Round {s['round']}/{s['max_rounds']}",
                f"Rounds remaining: {remaining}",
                "",
                "Resource pool (total 100 each):",
            ]
            for r, pool in resources.items():
                agent_d = draft.get(r, "not set")
                opp_d = opp.get(r, "?")
                opp_gets = pool - opp_d if isinstance(opp_d, int) else "?"
                lines.append(
                    f"  {r:10s}: pool={pool} | opp wants {opp_d} for themselves "
                    f"(leaves {opp_gets} for you) | your draft: {agent_d}"
                )
            lines.append(
                "\nActions: make_offer(resource, amount) | finalize_offer() | "
                "accept_opponent_offer()"
            )
            return "\n".join(lines)

        @self.mcp.tool()
        def make_offer(resource: str, amount: int, episode_id: str = "default") -> str:
            s = get_session(episode_id)
            if not s["current_resources"]: self._generate_negotiation(s)
            
            resources = s["current_resources"]
            if resource not in resources:
                valid = list(resources.keys())
                return f"Unknown resource '{resource}'. Valid: {valid}"
            if not (0 <= amount <= 100):
                return "Amount must be between 0 and 100."
            s["agent_share"][resource] = amount
            s["last_reward"] = -0.05
            filled = len(s["agent_share"])
            total = len(resources)
            return (
                f"Draft updated: {resource} = {amount}. "
                f"({filled}/{total} resources set). "
                f"Call finalize_offer() when ready."
            )

        @self.mcp.tool()
        def finalize_offer(episode_id: str = "default") -> str:
            s = get_session(episode_id)
            if not s["current_resources"]: self._generate_negotiation(s)
            
            resources = s["current_resources"]
            if len(s["agent_share"]) < len(resources):
                missing = [r for r in resources if r not in s["agent_share"]]
                return f"You must set all resources first. Missing: {missing}"

            s["round"] += 1
            s["step_count"] += 1
            proposal = s["agent_share"].copy()
            s["history"].append({"round": s["round"], "agent": proposal})

            if s["round"] > s["max_rounds"]:
                s["last_reward"] = -1.0
                s["failed_deals"] += 1
                return "No deal — rounds exhausted. Reward: -1.0."

            if self._check_opponent_accept(s, proposal):
                total_pool = sum(resources.values())
                agent_total = sum(proposal.values())
                share_quality = agent_total / total_pool
                reward = 1.5 + share_quality
                s["last_reward"] = reward
                s["deal_reached"] = True
                s["deals_made"] += 1
                if s["deals_made"] % 2 == 0:
                    s["difficulty"] = min(s["difficulty"] + 1, 10)
                self._generate_negotiation(s)
                return f"✅ DEAL REACHED! Reward: +{reward:.2f}. Next: Difficulty {s['difficulty']}."
            else:
                s["last_reward"] = -0.3
                greed = _OPPONENT_GREED.get(s["difficulty"], 0.5) * random.uniform(0.95, 1.0)
                s["opponent_offer"] = {r: int(100 * greed) for r in resources}
                return f"❌ Rejected. Opponent counter-offers. Round {s['round']}/{s['max_rounds']}."

        @self.mcp.tool()
        def accept_opponent_offer(episode_id: str = "default") -> str:
            s = get_session(episode_id)
            if not s["current_resources"]: self._generate_negotiation(s)
            resources = s["current_resources"]
            opp = s["opponent_offer"]
            agent_total = sum(100 - val for val in opp.values())
            share_quality = agent_total / (100 * len(resources))
            reward = 1.0 + share_quality
            s["last_reward"] = reward
            s["deals_made"] += 1
            if s["deals_made"] % 2 == 0:
                s["difficulty"] = min(s["difficulty"] + 1, 10)
            self._generate_negotiation(s)
            return f"✅ Accepted. Reward: +{reward:.2f}. Next: Difficulty {s['difficulty']}."

        @self.mcp.tool()
        def record_thought(thought: str, episode_id: str = "default") -> str:
            s = get_session(episode_id)
            if len(s["thoughts"]) < 3: s["last_reward"] = 0.1
            s["thoughts"].append(thought[:500])
            return f"Thought recorded ({len(s['thoughts'])}/3)."

    def _generate_negotiation(self, s: dict) -> None:
        d = s["difficulty"]
        n_resources = min(d, 5)
        resources = RESOURCE_NAMES[:n_resources]
        s["current_resources"] = {r: 100 for r in resources}
        s["agent_share"] = {}
        s["round"] = 0
        s["deal_reached"] = False
        s["opponent_offer"] = {r: int(100 * _OPPONENT_GREED.get(d, 0.5)) for r in resources}

    def _check_opponent_accept(self, s: dict, proposal: dict) -> bool:
        greed = _OPPONENT_GREED.get(s["difficulty"], 0.5)
        for r, pool in s["current_resources"].items():
            if (pool - proposal.get(r, 0)) < greed * pool: return False
        return True

    def reset(self, episode_id: Optional[str] = None, difficulty: Optional[int] = None, **kwargs) -> Observation:
        eid = episode_id or str(uuid4())
        s = get_session(eid)
        if difficulty is not None:
            s["difficulty"] = difficulty
        else:
            s["difficulty"] = 1
            s["deals_made"] = 0
        s["step_count"] = 0
        self._generate_negotiation(s)
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
