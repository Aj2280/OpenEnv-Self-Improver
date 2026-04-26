"""Math Escalation GRPO Trainer — Hugging Face Space (minimal Gradio UI).

Only **five** controls are exposed to avoid Gradio / Space iframe limits and
older-Gradio compatibility issues. Everything else uses defaults in
`train_worker.py`.
"""

import os
import subprocess
import threading
from pathlib import Path

import gradio as gr

LOG_FILE = Path("train.log")
STATUS_FILE = Path("status.txt")
RUN_LOCK = threading.Lock()
TRAINING_THREAD: threading.Thread | None = None
APP_VERSION = "params-ui-v2-five-hubfix"

# Five user-facing defaults (must match train_worker fallbacks where applicable).
DEFAULTS = {
    "model_name": "unsloth/Qwen2.5-0.5B-Instruct",
    "dataset_size": 200,
    "max_steps": 30,
    "learning_rate": 5e-6,
    "num_generations": 4,
}

MODEL_CHOICES = [
    "unsloth/Qwen2.5-0.5B-Instruct",
    "unsloth/Qwen2.5-1.5B-Instruct",
]


def _set_status(text: str) -> None:
    STATUS_FILE.write_text(text, encoding="utf-8")


def _read_tail(max_lines: int = 200) -> str:
    if not LOG_FILE.exists():
        return "No logs yet."
    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-max_lines:]
    return "\n".join(tail) if tail else "No logs yet."


def _format_config(cfg: dict) -> str:
    rows = "\n".join(f"  {k}: {v}" for k, v in cfg.items())
    return f"=== Run config (5 UI params; rest = worker defaults) ===\n{rows}\n"


def get_logs() -> str:
    status = STATUS_FILE.read_text(encoding="utf-8").strip() if STATUS_FILE.exists() else "idle"
    return f"App version: {APP_VERSION}\nStatus: {status}\n\n{_read_tail()}"


def _parse_lr(val) -> float:
    if val is None:
        return float(DEFAULTS["learning_rate"])
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().replace(",", ".")
    if not s:
        return float(DEFAULTS["learning_rate"])
    try:
        return float(s)
    except ValueError:
        return float(DEFAULTS["learning_rate"])


def _run_training_subprocess(cfg: dict) -> None:
    _set_status("running")
    env = os.environ.copy()
    # Only these five are user-controlled; train_worker uses defaults for the rest.
    env["MES_MODEL_NAME"] = str(cfg["model_name"])
    env["MES_DATASET_SIZE"] = str(int(cfg["dataset_size"]))
    env["MES_MAX_STEPS"] = str(int(cfg["max_steps"]))
    env["MES_LEARNING_RATE"] = str(float(cfg["learning_rate"]))
    env["MES_NUM_GENERATIONS"] = str(int(cfg["num_generations"]))

    with LOG_FILE.open("w", encoding="utf-8") as out:
        out.write(f"=== Starting background training ({APP_VERSION}) ===\n")
        out.write(_format_config(cfg))
        out.flush()
        try:
            subprocess.run(
                ["python", "train_worker.py"],
                check=True,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
            )
            out.write("\n=== Training process finished successfully ===\n")
            _set_status("success")
        except subprocess.CalledProcessError as e:
            out.write(f"\n=== Training process failed with exit code {e.returncode} ===\n")
            _set_status(f"failed (exit {e.returncode})")
        except Exception as e:
            out.write(f"\n=== Training process encountered an error: {e} ===\n")
            _set_status("failed (exception)")
        finally:
            out.flush()


def start_training(
    model_name: str,
    dataset_size: float,
    max_steps: float,
    learning_rate,
    num_generations: float,
) -> str:
    global TRAINING_THREAD
    cfg = {
        "model_name": model_name or DEFAULTS["model_name"],
        "dataset_size": int(dataset_size),
        "max_steps": int(max_steps),
        "learning_rate": _parse_lr(learning_rate),
        "num_generations": int(num_generations),
    }

    with RUN_LOCK:
        if TRAINING_THREAD and TRAINING_THREAD.is_alive():
            return (
                "Training is already running — wait for it to finish or rebuild the Space.\n\n"
                + get_logs()
            )
        LOG_FILE.write_text("", encoding="utf-8")
        _set_status("starting")
        TRAINING_THREAD = threading.Thread(
            target=_run_training_subprocess, args=(cfg,), daemon=True
        )
        TRAINING_THREAD.start()

    return get_logs()


def reset_defaults():
    return [
        DEFAULTS["model_name"],
        DEFAULTS["dataset_size"],
        DEFAULTS["max_steps"],
        str(DEFAULTS["learning_rate"]),
        DEFAULTS["num_generations"],
    ]


if not STATUS_FILE.exists():
    _set_status("idle — set the 5 options and click Start Training")
if not LOG_FILE.exists():
    LOG_FILE.write_text(
        "Math Escalation GRPO Trainer (5-parameter demo).\n"
        "Choose model, dataset size, steps, learning rate, and generations; then Start Training.\n",
        encoding="utf-8",
    )


with gr.Blocks(title="Math Escalation GRPO Trainer") as demo:
    gr.Markdown("# Math Escalation GRPO Trainer")
    gr.Markdown(
        "This Space uses a **small 5-control** form so it stays compatible with Hugging Face "
        "Spaces. Adjust the options, then click **Start Training**. Other hyperparameters "
        "(batch size, LoRA, temperature, etc.) use safe defaults in the training worker.\n\n"
        "**Note:** Transformers may print yellow `FutureWarning` lines during GRPO — those are "
        "normal. The bottom **Container** tray red error **BodyStreamBuffer was aborted** usually "
        "means the Space restarted or the log stream was cut off; use **Refresh logs** here for "
        "the real `train.log` tail.\n\n"
        "**Hub upload:** In Space **Settings → Secrets**, set **`HF_TOKEN`** to a token with "
        "**Write** permission (same account or org you want to publish under). Optional: "
        "**`HF_OUTPUT_REPO`** = `your-user/your-model-repo` to override the auto name "
        "(`math-escalation-grpo-<model>` under the token owner)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Training options (5)")
            model_name = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULTS["model_name"],
                label="Model",
            )
            dataset_size = gr.Slider(
                minimum=50,
                maximum=400,
                step=10,
                value=DEFAULTS["dataset_size"],
                label="Dataset size (problems)",
            )
            max_steps = gr.Slider(
                minimum=5,
                maximum=100,
                step=5,
                value=DEFAULTS["max_steps"],
                label="Max GRPO steps",
            )
            learning_rate = gr.Textbox(
                value=str(DEFAULTS["learning_rate"]),
                label="Learning rate (e.g. 5e-6 or 0.000005)",
            )
            num_generations = gr.Slider(
                minimum=2,
                maximum=6,
                step=1,
                value=DEFAULTS["num_generations"],
                label="GRPO generations per prompt",
            )

            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary")
                reset_btn = gr.Button("Reset to defaults")

        with gr.Column(scale=2):
            gr.Markdown("### Status and logs")
            logs = gr.Textbox(
                label="Status + train.log tail",
                value=get_logs(),
                lines=28,
            )
            refresh_btn = gr.Button("Refresh logs")

    inputs = [model_name, dataset_size, max_steps, learning_rate, num_generations]
    start_btn.click(fn=start_training, inputs=inputs, outputs=logs)
    refresh_btn.click(fn=get_logs, outputs=logs)
    reset_btn.click(fn=reset_defaults, outputs=inputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
