# 🚀 OpenEnv-Self-Improver
### *OpenEnv Theme #4: Recursive Skill Amplification*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.2-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Theme #4](https://img.shields.io/badge/Theme-%234%20Self--Improvement-orange)](#)
[![TRL](https://img.shields.io/badge/Trainer-TRL%20GRPO-green)](https://huggingface.co/docs/trl)
[![Unsloth](https://img.shields.io/badge/Inference-Unsloth-purple)](https://github.com/unslothai/unsloth)
[![Environments](https://img.shields.io/badge/Environments-3-red)](#-three-environments)

> **GitHub Repo:** [https://github.com/Aj2280/OpenEnv-Self-Improver](https://github.com/Aj2280/OpenEnv-Self-Improver)  
> **Hugging Face Space:** [https://huggingface.co/spaces/Abhi2280/OpenEnv-Self-Improver](https://huggingface.co/spaces/Abhi2280/OpenEnv-Self-Improver) (Pending Rename)  
> **Training Notebook:** [Colab Notebook](TRAINING.md)

---

## 🎯 The Problem

Static benchmarks cannot test *recursive self-improvement* — the ability of an agent to grow into harder tasks as it masters easier ones. Most RL environments present a fixed difficulty, meaning reward signal dies once the agent is competent.

This suite solves this with **three distinct adaptive curriculum environments** built on OpenEnv, each implementing a different form of self-improvement:

| Form of Self-Improvement | Environment |
|--------------------------|-------------|
| **Recursive reasoning** — solve increasingly complex math | 🧮 Math Escalation |
| **Strategic negotiation** — out-negotiate a tougher opponent each round | 🤝 Negotiation Arena |
| **Algorithmic coding** — write code for increasingly hard programming challenges | 💻 Coding Competition |

## 🌟 Key Features

- **Recursive RL Pipeline (GRPO)**: Integrates with Unsloth and TRL for high-efficiency training on consumer GPUs.
- **Deterministic Verification**: No "vibes" based rewards. All success metrics are programmatically verified.
- **Self-Improving Rewards**: Reward functions are dense and shape reasoning behavior (<thought> tags) and formatting.
- **Unified FastAPI Backend**: A single server provides all environments via the OpenEnv standard.

## 🚀 Getting Started

### 1. Installation
```bash
uv sync
```

### 2. Run the Environment Server
```bash
uv run --project . server
```

### 3. Training (Math Escalation)
Open `train_math_improved.ipynb` in Google Colab (T4 GPU recommended) to start the recursive RL training loop.

## 🏆 Hackathon Themes Satisfied
- **Theme #4: Self-Improvement**: Core architecture revolves around agents improving their own internal capabilities to solve escalating tiers.
- **Environmental Robustness**: Multi-worker session isolation and concurrent rollout support.
