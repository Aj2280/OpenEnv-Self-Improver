"""
Math Escalation — RL Training Simulation (Theme #4: Self-Improvement)

This script demonstrates the full RL loop:
  1. Environment emits an observation (math problem)
  2. Agent reasons and proposes an action (numeric answer)
  3. Environment returns a multi-component reward
  4. Reward history is tracked and plotted

In a real TRL/Unsloth run, step (2) would be replaced by LLM sampling
and GRPO would update the model weights. This script proves the environment
produces clean, non-zero reward signals that RL can learn from.

Guideline coverage:
  ✅ #1  Task has crisp verification (exact numeric match)
  ✅ #2  Complete RL loop (observe → act → reward → log)
  ✅ #4  reset() / step() / state / reward all implemented
  ✅ #6  Curriculum: starts easy, escalates only on success
  ✅ #7  Multi-component reward (correct + format + level-clear bonus)
  ✅ #8  Anti-hack: step budget, hint penalty, thought-spam cap
  ✅ #9  Process feedback: record_thought() earns small reward
  ✅ #11 Verifiable reward (no learned reward model needed)
"""

import requests
import re
import math
import json
import time
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"
EPISODE_STEPS = 60   # per episode
N_EPISODES = 3       # multiple episodes to show curriculum reset


# ─────────────────────────────────────────────────────────────────────────────
# Oracle solver — simulates what a trained LLM would compute
# In real RL: this is replaced by model.generate(prompt)
# ─────────────────────────────────────────────────────────────────────────────

def agent_solve(problem_str: str) -> float:
    """
    Rule-based oracle solver. Handles all 10 difficulty tiers.
    Returns the correct float answer.
    """
    # Strip the prefix: "[Difficulty N/10] Solve: <expr>"
    parts = problem_str.split("Solve: ", 1)
    expr = parts[1].strip() if len(parts) > 1 else problem_str

    # Tier 8: "Solve for x: 8x + 33 = 57"
    m = re.search(r"(\d+)x \+ (-?\d+) = (\d+)", expr)
    if m:
        a, b, c = map(float, m.groups())
        return (c - b) / a

    # Tier 10: "Solve for x: (2x + 10) / 4 = 5"
    m = re.search(r"\((\d+)x \+ (-?\d+)\) / (\d+) = (\d+)", expr)
    if m:
        a, b, d, c = map(float, m.groups())
        return (c * d - b) / a

    # Tier 9: "sqrt(N)"
    m = re.search(r"sqrt\((\d+)\)", expr)
    if m:
        return float(math.sqrt(int(m.group(1))))

    # Tiers 1-7: arithmetic expressions — safe eval on cleaned string
    # Remove anything that isn't a number or operator
    clean = re.sub(r"[^0-9+\-*/().² ]", "", expr).strip()
    try:
        return float(eval(clean))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers
# ─────────────────────────────────────────────────────────────────────────────

# Global session state
CURRENT_EPISODE_ID = None

def call(tool: str, args: dict = None) -> dict:
    """Helper to call an MCP tool via the OpenEnv step endpoint."""
    try:
        actual_args = args or {}
        if CURRENT_EPISODE_ID:
            actual_args["episode_id"] = CURRENT_EPISODE_ID
            
        payload = {
            "action": {"type": "call_tool", "tool_name": tool, "arguments": actual_args}
        }
        if CURRENT_EPISODE_ID:
            payload["episode_id"] = CURRENT_EPISODE_ID
            
        resp = requests.post(f"{BASE_URL}/step", json=payload)
        return resp.json()
    except Exception as e:
        print(f"[ERROR] Tool call failed: {e}")
        return {}

def reset_env():
    """Reset the environment to a clean state."""
    global CURRENT_EPISODE_ID
    try:
        resp = requests.post(f"{BASE_URL}/reset", json={})
        data = resp.json()
        # Capture episode_id robustly
        meta = data.get("metadata", {}) or {}
        obs  = data.get("observation", {}) or {}
        
        CURRENT_EPISODE_ID = meta.get("episode_id") or obs.get("episode_id") or data.get("episode_id")
        return data
    except Exception as e:
        print(f"[ERROR] Reset failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(episode_num: int) -> list[dict]:
    """
    Run one full episode. Returns list of step records for reward plotting.
    """
    print(f"\n{'='*60}")
    print(f"  Episode {episode_num + 1}/{N_EPISODES}")
    print(f"{'='*60}")

    reset_env()
    records = []
    done = False
    step = 0
    cumulative_reward = 0.0

    while not done and step < EPISODE_STEPS:
        # ── Observe ────────────────────────────────────────────────────
        get_resp = call("get_problem")
        obs_data = get_resp.get("observation", {})
        result_obj = obs_data.get("result", {}) if isinstance(obs_data, dict) else {}
        problem_str = result_obj.get("data", "") if isinstance(result_obj, dict) else str(obs_data)
        
        # Difficulty extraction with fallback
        difficulty = get_resp.get("metadata", {}).get("difficulty")
        if difficulty is None:
             # Try to extract from result string prefix "[Difficulty N/10]"
             m = re.search(r"\[Difficulty (\d+)/10\]", problem_str)
             difficulty = int(m.group(1)) if m else 1

        if not problem_str:
            print("[WARN] Empty problem string, skipping step.")
            step += 1
            continue

        # ── Chain-of-thought (process reward, Theme #4 core) ──────────
        thought = (
            f"Tier {difficulty}: I need to parse '{problem_str}' and "
            f"apply the correct math strategy."
        )
        call("record_thought", {"thought": thought})

        # ── Act ────────────────────────────────────────────────────────
        answer = agent_solve(problem_str)

        # ── Submit ─────────────────────────────────────────────────────
        sub_resp = call("submit_answer", {"answer": answer})
        reward   = sub_resp.get("reward", 0.0)
        done     = sub_resp.get("done", False)
        # Extract metadata and result from top-level and observation
        meta     = sub_resp.get("metadata", {})
        obs_sub  = sub_resp.get("observation", {})
        res_obj  = obs_sub.get("result", {}) if isinstance(obs_sub, dict) else {}
        result   = res_obj.get("data", "") if isinstance(res_obj, dict) else str(obs_sub)

        cumulative_reward += reward
        records.append({
            "episode":    episode_num,
            "step":       step,
            "difficulty": difficulty,
            "answer":     answer,
            "reward":     reward,
            "cumulative": cumulative_reward,
            "done":       done,
        })

        status_icon = "✅" if reward > 0 else "❌"
        print(
            f"  [{step:>3}] Tier {difficulty} | "
            f"ans={answer:<8.2f} | r={reward:+.2f} | Σ={cumulative_reward:+.2f} | "
            f"{status_icon} {result[:50]}"
        )

        step += 1

    final_difficulty = meta.get("difficulty", "?")
    print(f"\n  Episode done. Final difficulty reached: {final_difficulty}/10")
    print(f"  Cumulative reward: {cumulative_reward:+.2f} | Steps: {step}")

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Plot generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_plots(all_records: list[dict]):
    """Generate reward curve plots. Falls back gracefully if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        plots_dir = Path(__file__).parent / "plots"
        plots_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#1a1a2e")
        colors = ["#e94560", "#16213e", "#0f3460", "#533483"]
        accent = "#e94560"

        # ── Plot 1: Per-step reward per episode ──────────────────────
        ax1 = axes[0]
        ax1.set_facecolor("#16213e")
        for ep in range(N_EPISODES):
            ep_recs = [r for r in all_records if r["episode"] == ep]
            steps   = [r["step"] for r in ep_recs]
            rewards = [r["reward"] for r in ep_recs]
            c = colors[ep % len(colors)]
            ax1.plot(steps, rewards, color=c, alpha=0.8, lw=1.5, label=f"Episode {ep+1}")
            ax1.fill_between(steps, 0, rewards, color=c, alpha=0.15)

        ax1.axhline(0, color="white", lw=0.5, ls="--", alpha=0.3)
        ax1.set_title("Step Reward per Episode", color="white", fontsize=13, pad=10)
        ax1.set_xlabel("Step", color="white")
        ax1.set_ylabel("Reward", color="white")
        ax1.tick_params(colors="white")
        ax1.spines["bottom"].set_color("#444")
        ax1.spines["left"].set_color("#444")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#444")

        # ── Plot 2: Cumulative reward per episode ────────────────────
        ax2 = axes[1]
        ax2.set_facecolor("#16213e")
        for ep in range(N_EPISODES):
            ep_recs = [r for r in all_records if r["episode"] == ep]
            steps   = [r["step"] for r in ep_recs]
            cumuls  = [r["cumulative"] for r in ep_recs]
            c = colors[ep % len(colors)]
            ax2.plot(steps, cumuls, color=c, alpha=0.9, lw=2, label=f"Episode {ep+1}")

        ax2.set_title("Cumulative Reward per Episode", color="white", fontsize=13, pad=10)
        ax2.set_xlabel("Step", color="white")
        ax2.set_ylabel("Cumulative Reward", color="white")
        ax2.tick_params(colors="white")
        ax2.spines["bottom"].set_color("#444")
        ax2.spines["left"].set_color("#444")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.legend(facecolor="#1a1a2e", labelcolor="white", edgecolor="#444")

        fig.suptitle(
            "Math Escalation Environment — RL Reward Curves\n(Theme #4: Self-Improvement)",
            color="white", fontsize=14, y=1.02
        )
        plt.tight_layout()
        out = plots_dir / "reward_curve.png"
        fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"\n📊 Reward curve saved → {out}")
        return str(out)
    except ImportError:
        print("\n[INFO] matplotlib not installed — skipping plot generation.")
        print("       Install with: pip install matplotlib")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("🚀 Math Escalation — RL Loop Demo (Theme #4: Self-Improvement)")
    print("   Guidelines covered: #1 #2 #4 #6 #7 #8 #9 #11")
    print(f"   Server: {BASE_URL}")

    # Verify server is up
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=3)
        print(f"   Health: {r.status_code}")
    except Exception:
        try:
            reset_env()
            print("   Server reachable via /reset.")
        except Exception as e:
            print(f"\n❌ Server not reachable at {BASE_URL}. Start it first:")
            print("   uv run --project . server")
            sys.exit(1)

    # Run episodes
    all_records: list[dict] = []
    for ep in range(N_EPISODES):
        records = run_episode(ep)
        all_records.extend(records)
        time.sleep(0.5)  # brief pause between episodes

    # Summary statistics
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    total_reward = sum(r["reward"] for r in all_records)
    correct = sum(1 for r in all_records if r["reward"] > 0)
    total   = len(all_records)
    max_diff = max(r["difficulty"] for r in all_records)

    print(f"  Total steps    : {total}")
    print(f"  Correct answers: {correct}/{total} ({100*correct//total}%)")
    print(f"  Total reward   : {total_reward:+.2f}")
    print(f"  Peak difficulty: {max_diff}/10")

    # Save raw data for reference
    data_path = Path(__file__).parent / "plots" / "training_data.json"
    data_path.parent.mkdir(exist_ok=True)
    with open(data_path, "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"  Data saved     : {data_path}")

    # Generate plots
    generate_plots(all_records)

    if max_diff > 10:
        print("\n🏆 SUCCESS — Agent cleared all 10 difficulty tiers!")
    else:
        print(f"\n📈 Agent reached tier {max_diff}/10. More training needed for full mastery.")

    print("\nNext step: Run the TRL/Unsloth Colab notebook to train a real LLM")
    print("           on this environment. See TRAINING.md for instructions.")


if __name__ == "__main__":
    main()
