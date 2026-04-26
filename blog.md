# Math Escalation: How I Taught a Tiny LLM to Do Math (and What I Learned Along the Way)

*A story about curriculum learning, reinforcement training, broken tokens, and one stubborn 0.5B model.*

---

## It Started With a Simple Question

Can a small language model — one that fits on a free GPU — actually learn to do math better?

Not GPT-4. Not Claude. A **0.5 billion parameter model**. The kind that hallucinates multiplication tables and confidently tells you that 7 × 8 = 54.

That question became **Math Escalation** — a full end-to-end AI training pipeline I built to train, evaluate, and improve a small LLM on math reasoning using **curriculum learning** and **GRPO reinforcement learning**.

This is the story of how I built it, what broke, what worked, and what I learned.

---

## The Problem: Small LLMs Are Bad at Math

If you've ever asked a tiny language model to solve `(3 + 5) × 2 - 4 ÷ 2`, you know the pain. It might give you a confident wrong answer. It might ignore order of operations. It might even skip showing any work at all.

The core issues are:
- **Carry errors** in multi-digit arithmetic
- **Order of operations** confusion
- **Equation solving** falls apart beyond simple cases
- **Final answer extraction** — the model solves it correctly but then writes the wrong number at the end

These aren't knowledge gaps. The model *knows* math. It just doesn't *reason* through it reliably. That's the distinction that drove this entire project.

---

## The Idea: Teach It Like a Student

The insight behind Math Escalation is simple: **don't throw hard problems at an untrained model**.

Instead, build a curriculum. Start with easy addition. Move to subtraction. Then multiplication. Then order of operations. Then algebra. Then equations. Like a student working through a textbook, the model should earn its way to harder problems by getting easier ones right first.

This is **curriculum learning** — and it's been shown to improve training stability and final performance in reinforcement learning settings.

Combined with **GRPO (Group Relative Policy Optimization)** — a reinforcement learning algorithm that rewards correct reasoning — the idea was: *give the model structured feedback on its math answers, and let it figure out how to improve*.

---

## Building the System

### The Dataset: 200 Problems Across 10 Tiers

The first piece was the curriculum dataset. I built a generator that creates math problems across **10 difficulty tiers**:

- **Tier 1–2:** Simple addition and subtraction (`12 + 7`, `45 - 18`)
- **Tier 3–4:** Multiplication and division
- **Tier 5–6:** Order of operations (`(3 + 5) × 2`)
- **Tier 7–8:** Square roots and mixed expressions
- **Tier 9–10:** Linear equations and multi-step algebra

Each run builds ~200 problems spread across tiers. The model trains on these in order, receiving reward signals based on whether it got the answer right.

### The Training Loop: GRPO with Reward Shaping

The training uses **TRL's GRPO trainer** with **Unsloth 4-bit optimization** on top of `Qwen2.5-0.5B-Instruct`. The reward function scores each completion on three things:

1. **Correct numeric answer** — did the final number match the expected answer?
2. **Valid reasoning format** — did it use `<thought>` tags to show its work?
3. **Presence of a final output** — did it actually give a number at the end?

This three-part reward structure means the model isn't just incentivized to be right — it's incentivized to *show its work* and *be explicit about its answer*.

### The Architecture: End-to-End

```
User / Frontend (React + Vite)
        ↓
FastAPI Math Environment
        ↓
Training Worker
        ↓
GRPO Trainer (TRL + Unsloth)
        ↓
Fine-tuned LoRA Adapters
        ↓
Local Evaluation (100 problems)
        ↓
Hugging Face Hub
```

The full system includes a **React + Vite frontend** for configuring runs, a **FastAPI backend** for managing the math environment, and a **Hugging Face Space** that runs the actual training on a free T4 GPU.

---

## What Actually Happened During Training

Here's where the story gets honest.

### Run 1: 5 Steps

The first run completed in about 2 minutes. Scores looked like this:

| Step | Reward |
|------|--------|
| 1 | 0.55 |
| 2 | -0.01 |
| 3 | 0.175 |
| 4 | 0.175 |
| 5 | 0.55 |

**Average: ~28.7%**

Interesting — but something was off. Every single completion had `clipped_ratio: 1.0`. That means every single generation hit the 128-token limit and got cut off. The model never finished a single answer.

That's a problem. You can't reward correct reasoning if the reasoning is always truncated.

### Run 2: 20 Steps

I ran it longer. Surely more steps = better scores, right?

**Average: ~15.3%**

Worse. The score actually dropped. More training with a broken generation length just made things more unstable.

### Run 3: Back to 5 Steps

Same as Run 1. Same scores. The system was consistent — consistently constrained by `max_completion: 128`.

### The Real Lesson

The bottleneck wasn't steps. It wasn't learning rate. It wasn't the dataset.

It was `max_completion: 128`. Every answer was being cut off before the model could finish its thought. The fix is simple — increase it to 512 or 1024 — but it was a powerful reminder that **hyperparameter blindspots can completely mask whether your model is actually learning**.

---

## The Upload Saga

Training worked. Uploading didn't. Here's the journey:

**Attempt 1:** `403 Forbidden — You don't have the rights to create a model under namespace "Abhi2280".`

The token had read-only access. Classic.

**Attempt 2 (after creating the repo manually):** `403 Forbidden — pass create_pr=1 as a query parameter to create a Pull Request.`

The repo existed now, but the token still couldn't write to `main`. The fix: update the token permissions to **Write** at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

Every error was a lesson in how HuggingFace's permission model works — and why your Space secrets and token scopes matter as much as your training code.

---

## The Evaluation System

Separate from training, Math Escalation includes a **local evaluation script** that tests the model on 100 deterministic math problems. This gives a clean accuracy score before and after training — the kind of number you can actually point to and say "we improved this."

The evaluation uses:
- **Prompt engineering** to format questions consistently
- **Answer extraction** with regex to pull the final number from messy outputs
- **Self-consistency sampling** to pick the most common answer across multiple generations

This is important because the training reward and the evaluation metric are measuring slightly different things — and that gap tells you a lot about whether the model is genuinely improving or just gaming the reward.

---

## The Full Tech Stack

| Layer | Technology |
|-------|------------|
| Base Model | Qwen2.5-0.5B-Instruct |
| Optimization | Unsloth 4-bit |
| Training | TRL / GRPO |
| Backend | FastAPI |
| Frontend | React + Vite |
| Deployment | Hugging Face Spaces |
| Model Storage | Hugging Face Hub |
| Hardware | Tesla T4 (free tier) |
| Language | Python + JavaScript |

---

## What I'd Do Differently

1. **`max_completion: 512` from day one.** This single change would have made every run more meaningful.
2. **More steps, more data.** 5–20 steps on 200 problems is a proof of concept. Real improvement needs 100+ steps on 1000+ problems.
3. **Log intermediate completions.** Seeing *what the model actually writes* during training tells you so much more than the reward score alone.
4. **Separate reward components.** Log format reward, correctness reward, and termination reward separately so you know which part is working.
5. **Set up the HF token before training, not after.** Every developer learns this the hard way.

---

## What This Project Demonstrates

Math Escalation isn't just about math. It's a complete pattern for **end-to-end LLM improvement**:

- **Curriculum data generation** → structured learning signal
- **GRPO reinforcement learning** → reward-based improvement without human labels
- **Chain-of-thought prompting** → explicit reasoning traces
- **Local evaluation** → measurable before/after comparison
- **Cloud deployment** → reproducible, shareable training runs
- **Hub integration** → versioned model artifacts

This same architecture — swap out the math problems for code, legal reasoning, or medical QA — and you have a general-purpose RLHF-lite pipeline for any domain.

---

## Use Cases Beyond This Project

- **Educational AI tutors** that reason through problems step by step
- **Domain-specific fine-tuning** for small models in low-resource environments
- **Benchmarking RL methods** on small models before scaling up
- **Teaching RL concepts** in an ML course with a concrete, runnable example
- **Portfolio project** demonstrating training, evaluation, deployment, and frontend integration

---

## Conclusion: The Model Is Still Learning. So Am I.

Math Escalation started as a question — *can a tiny model get better at math?* — and became a lesson in systems thinking, debugging pipelines, and the gap between \"training ran\" and \"training worked.\"

The reward scores were modest. The model never finished a completion during these runs. The token permissions broke twice. And yet — the pipeline works. The curriculum generates. The reward function fires. The evaluation runs. The Hub (almost) receives the model.

That's the thing about end-to-end ML projects: **every broken piece is telling you something**. The 403 error taught me about HF permissions. The clipped completions taught me about generation length. The flat reward curve taught me that more steps don't fix architecture problems.

The next run will have `max_completion: 512`. And the one after that will have 500 steps. And somewhere in there, a 0.5B model is going to solve a linear equation it couldn't solve before.

That's the escalation.

---

*Built with Unsloth, TRL, FastAPI, React, and a lot of patience.*  
*Deployed on Hugging Face Spaces. Trained on a free T4 GPU.*  
*Model: [Abhi2280/math-escalation-grpo-qwen2-5-0-5b-instruct](https://huggingface.co/Abhi2280/math-escalation-grpo-qwen2-5-0-5b-instruct)*

---

**Tags:** `#reinforcement-learning` `#GRPO` `#math-reasoning` `#Unsloth` `#Qwen` `#fine-tuning` `#curriculum-learning` `#LLM` `#hackathon`

