# 🧮 Self-Improvement Environment Suite
### *OpenEnv Theme #4: Recursive Skill Amplification*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.2-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Theme #4](https://img.shields.io/badge/Theme-%234%20Self--Improvement-orange)](#)
[![TRL](https://img.shields.io/badge/Trainer-TRL%20GRPO-green)](https://huggingface.co/docs/trl)
[![Unsloth](https://img.shields.io/badge/Inference-Unsloth-purple)](https://github.com/unslothai/unsloth)
[![Environments](https://img.shields.io/badge/Environments-3-red)](#-three-environments)

> **Hugging Face Space:** [YOUR_HF_SPACE_LINK]  
> **Training Notebook:** [Colab Notebook](TRAINING.md)  
> **Video Demo:** [YOUR_VIDEO_LINK]

---

## 🎯 The Problem

Static benchmarks cannot test *recursive self-improvement* — the ability of an agent to grow into harder tasks as it masters easier ones. Most RL environments present a fixed difficulty, meaning reward signal dies once the agent is competent.

This suite solves this with **three distinct adaptive curriculum environments** built on OpenEnv, each implementing a different form of self-improvement:

| Form of Self-Improvement | Environment |
|--------------------------|-------------|
| **Recursive reasoning** — solve increasingly complex math | 🧮 Math Escalation |
| **Strategic negotiation** — out-negotiate a tougher opponent each round | 🤝 Negotiation Arena |
| **Algorithmic coding** — write code for increasingly hard programming challenges | 💻 Coding Competition |

---

## 🌍 Three Environments

### 🧮 1. Math Escalation (`/`)
*Adaptive arithmetic → algebra curriculum*

| Tier | Challenge | Example |
|------|-----------|---------|
| 1 | Single-digit addition | `5 + 4` |
| 2 | Double-digit addition | `42 + 87` |
| 3 | Subtraction | `99 - 45` |
| 4 | Multiplication | `8 * 7` |
| 5 | Mixed ops | `3 * 4 + 10` |
| 6 | Nested ops | `2 * (5 + 8)` |
| 7 | Division | `72 / 9` |
| 8 | Linear equations | `Solve: 2x + 4 = 10` |
| 9 | Square roots | `sqrt(144)` |
| 10 | Multi-step algebra | `Solve: (2x + 10) / 4 = 5` |

**Tools:** `get_problem` · `submit_answer` · `get_hint` · `record_thought` · `get_status`  
**Reward:** +1.1 correct · +0.5 level-clear bonus · -0.5 wrong · -0.2 hint

---

### 🤝 2. Negotiation Arena (`/negotiate`)
*Self-play resource negotiation vs. rule-based opponent*

The agent negotiates over shared resources (food, water, shelter, tools, fuel). At each difficulty tier, the opponent becomes **greedier** (wants more for itself):

| Tier | Opponent Demand | Resources |
|------|----------------|-----------|
| 1 | 30% | food only |
| 3 | 40% | food, water, shelter |
| 5 | 50% | 4 resources (equal split) |
| 8 | 65% | 5 resources |
| 10 | 75% | 5 resources (very tough) |

The agent must learn to make strategic offers that the opponent accepts while maximizing its own share.

**Tools:** `get_negotiation_state` · `make_offer` · `finalize_offer` · `accept_opponent_offer` · `record_thought`  
**Reward:** +1.5 deal reached · +share_quality bonus · -0.3 per rejected offer · -1.0 no deal

---

### 💻 3. Coding Competition (`/code`)
*Evolving algorithmic challenges with sandboxed execution*

The agent writes Python functions that are run against hidden test cases:

| Tier | Challenge | Skill |
|------|-----------|-------|
| 1 | Double a number | Basic functions |
| 2 | Sum a list | Iteration |
| 3 | Find maximum | Comparison |
| 4 | Filter evens | List comprehension |
| 5 | Reverse string | Slicing |
| 6 | Check palindrome | String ops |
| 7 | Binary search | Algorithms |
| 8 | Word frequency | Dict operations |
| 9 | Flatten list | Nested iteration |
| 10 | Find primes | Number theory |

**Anti-hack:** sandboxed `exec()` — no `__import__`, `os`, `sys`, `subprocess`. Code capped at 1000 chars.

**Tools:** `get_challenge` · `submit_code` · `get_hint` · `record_thought` · `get_status`  
**Reward:** +1.5 all tests pass · partial credit · -0.3 error · +0.5 level-clear

---

## 🔄 The Self-Improvement Loop

```
┌─────────────────────────────────────────────────────────┐
│          RL Training Loop (TRL GRPO + Unsloth)          │
│                                                         │
│  1. Observe    →  get_problem / get_challenge / ...     │
│  2. Reason     →  record_thought("strategy...")         │
│  3. Act        →  submit_answer / submit_code / ...     │
│  4. Reward     →  multi-component signal returned       │
│  5. Escalate   →  difficulty++ on mastery               │
│  6. Update     →  GRPO shifts weights toward +reward    │
└─────────────────────────────────────────────────────────┘
```

Each tier only unlocks when the agent **demonstrates mastery** (2 consecutive correct answers/deals/code submissions). This forces recursive skill growth — the agent *cannot* stay comfortable at an easy level.

---

## 💰 Multi-Component Reward Design (Guideline #7)

Using **6+ independent reward signals** per environment prevents reward hacking:

```
Math:        correct(+1.0) + format(+0.1) + level_clear(+0.5) - wrong(-0.5) - hint(-0.2) + thought(+0.01)
Negotiate:   deal(+1.5) + share_quality(+0.0→+1.0) - rejected(-0.3) - no_deal(-1.0) + thought(+0.1)
Code:        all_pass(+1.5) + partial_credit - errors(-0.3) - failed_tests(-0.1) + level_clear(+0.5)
```

An agent cannot maximize one signal without genuinely solving the underlying task.

---

## 🛡️ Anti-Reward-Hacking (Guideline #8)

| Protection | Math | Negotiate | Code |
|------------|------|-----------|------|
| Step budget (hard cap) | 200 | — | 100 |
| Hint penalties | ✅ -0.2 | — | ✅ -0.2 |
| Thought-spam cap | 5/ep | 3/ep | 5/ep |
| Round limit | — | 15 rounds | — |
| Sandboxed execution | — | — | ✅ restricted exec |
| Forbidden patterns | — | — | ✅ no imports/os/sys |

---

## 📊 Training Results

![Reward Curve](plots/reward_curve.png)
*Oracle agent verification run: 60/60 correct answers, +81.00 cumulative reward, consistent across 3 episodes. The alternating reward pattern (1.1 / 1.6) shows tier-clear bonuses firing correctly.*

---

## 🚀 Getting Started

### Run All Three Environments

```bash
# 1. Install dependencies
uv sync

# 2. Start the server (all 3 environments on port 8000)
uv run --project . server

# Endpoints:
#   Math Escalation:   http://localhost:8000/
#   Negotiation Arena: http://localhost:8000/negotiate/
#   Coding Competition: http://localhost:8000/code/
#   API Docs (Math):   http://localhost:8000/docs
#   API Docs (Neg.):   http://localhost:8000/negotiate/docs
#   API Docs (Code):   http://localhost:8000/code/docs
```

### Run the RL Simulation

```bash
# Runs the oracle agent demo across all 3 environments
uv run python train.py
```

### Play Manually

```bash
# Interactive client (Math environment)
uv run python math_client.py --interactive
```

### Deploy to Hugging Face Spaces

```bash
git init && git add . && git commit -m "Self-Improvement Suite v0.2.0"
huggingface-cli repo create self-improvement-env --type space --sdk docker
git push https://huggingface.co/spaces/YOUR_NAME/self-improvement-env main
```

---

## 📁 Project Structure

```
math_escalation/
├── server/
│   ├── app.py                     # Multi-env FastAPI app (3 environments)
│   ├── math_environment.py        # 🧮 Math Escalation (10 tiers)
│   ├── negotiation_environment.py # 🤝 Negotiation Arena (10 tiers)
│   ├── coding_environment.py      # 💻 Coding Competition (10 challenges)
│   └── Dockerfile                 # HF Spaces deployment
├── train.py                       # RL loop demo + reward curve generation
├── math_client.py                 # Interactive manual play client
├── TRAINING.md                    # TRL/GRPO/Unsloth training guide
├── openenv.yaml                   # OpenEnv manifest
├── pyproject.toml                 # Package config
└── plots/
    ├── reward_curve.png            # Training evidence ✅
    └── training_data.json          # Raw episode data
```

---

## 🔬 RL Design Principles Applied

| Guideline | Implementation |
|-----------|----------------|
| #1 Crisp verification | Math: exact match · Code: test cases · Negotiate: deal/no-deal |
| #4 Env-first design | `reset()` / `step()` / `state` / reward explicit in all 3 envs |
| #6 Start simple | Math: `5+4` · Negotiate: 1 resource, 30% greedy · Code: `return n*2` |
| #7 Multi-component reward | 6+ independent signals per environment |
| #8 Anti-hacking | Step budgets, hint penalties, thought caps, sandboxed exec |
| #9 Process feedback | `record_thought()` in all 3 environments |
| #10 TRL + Unsloth | See TRAINING.md |
| #11 GRPO/RLVR | Verifiable rewards — no learned reward model needed |

---

*Built for the **Meta OpenEnv Hackathon 2026** — Theme #4: Self-Improvement*
