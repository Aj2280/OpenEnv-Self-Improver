"""Math Escalation GRPO Trainer — Gradio app for Hugging Face Spaces.

The user picks all hyperparameters from the UI and clicks **Start Training**.
Values are passed to `train_worker.py` as `MES_*` environment variables.
Auto-start is disabled so a fresh container does not silently burn GPU time
on a stale config.
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
APP_VERSION = "params-ui-v1"

# Default hyperparameters — match the demo defaults baked into train_worker.py.
DEFAULTS = {
    "model_name": "unsloth/Qwen2.5-0.5B-Instruct",
    "dataset_size": 400,
    "max_steps": 50,
    "learning_rate": 5e-6,
    "batch_size": 2,
    "grad_accum": 4,
    "num_generations": 4,
    "max_completion": 128,
    "temperature": 0.9,
    "lora_r": 16,
    "lora_alpha": 32,
    "warmup_ratio": 0.05,
}

MODEL_CHOICES = [
    "unsloth/Qwen2.5-0.5B-Instruct",
    "unsloth/Qwen2.5-1.5B-Instruct",
    "unsloth/Qwen2.5-3B-Instruct",
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
    return f"=== Run config ===\n{rows}\n"


def get_logs() -> str:
    status = STATUS_FILE.read_text(encoding="utf-8").strip() if STATUS_FILE.exists() else "idle"
    return f"App version: {APP_VERSION}\nStatus: {status}\n\n{_read_tail()}"


def _run_training_subprocess(cfg: dict) -> None:
    _set_status("running")
    env = os.environ.copy()
    env.update({
        "MES_MODEL_NAME":      str(cfg["model_name"]),
        "MES_DATASET_SIZE":    str(int(cfg["dataset_size"])),
        "MES_MAX_STEPS":       str(int(cfg["max_steps"])),
        "MES_LEARNING_RATE":   str(float(cfg["learning_rate"])),
        "MES_BATCH_SIZE":      str(int(cfg["batch_size"])),
        "MES_GRAD_ACCUM":      str(int(cfg["grad_accum"])),
        "MES_NUM_GENERATIONS": str(int(cfg["num_generations"])),
        "MES_MAX_COMPLETION":  str(int(cfg["max_completion"])),
        "MES_TEMPERATURE":     str(float(cfg["temperature"])),
        "MES_LORA_R":          str(int(cfg["lora_r"])),
        "MES_LORA_ALPHA":      str(int(cfg["lora_alpha"])),
        "MES_WARMUP_RATIO":    str(float(cfg["warmup_ratio"])),
    })

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
    learning_rate: float,
    batch_size: float,
    grad_accum: float,
    num_generations: float,
    max_completion: float,
    temperature: float,
    lora_r: float,
    lora_alpha: float,
    warmup_ratio: float,
) -> str:
    global TRAINING_THREAD
    cfg = {
        "model_name": model_name,
        "dataset_size": int(dataset_size),
        "max_steps": int(max_steps),
        "learning_rate": float(learning_rate),
        "batch_size": int(batch_size),
        "grad_accum": int(grad_accum),
        "num_generations": int(num_generations),
        "max_completion": int(max_completion),
        "temperature": float(temperature),
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "warmup_ratio": float(warmup_ratio),
    }

    with RUN_LOCK:
        if TRAINING_THREAD and TRAINING_THREAD.is_alive():
            return (
                "Training is already running — wait for it to finish or rebuild the Space "
                "to cancel.\n\n" + get_logs()
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
        DEFAULTS["learning_rate"],
        DEFAULTS["batch_size"],
        DEFAULTS["grad_accum"],
        DEFAULTS["num_generations"],
        DEFAULTS["max_completion"],
        DEFAULTS["temperature"],
        DEFAULTS["lora_r"],
        DEFAULTS["lora_alpha"],
        DEFAULTS["warmup_ratio"],
    ]


# Initialize idle status on cold start; do NOT auto-launch training.
if not STATUS_FILE.exists():
    _set_status("idle (configure parameters and click Start Training)")
if not LOG_FILE.exists():
    LOG_FILE.write_text(
        "Welcome to Math Escalation GRPO Trainer.\n"
        "Pick your hyperparameters on the left and click Start Training.\n",
        encoding="utf-8",
    )


with gr.Blocks(title="Math Escalation GRPO Trainer") as demo:
    gr.Markdown("# Math Escalation GRPO Trainer")
    gr.Markdown(
        "Configure GRPO hyperparameters below and click **Start Training**. "
        "Logs stream into the panel on the right. Use **Refresh Logs** to poll "
        "the latest output, and **Reset to Defaults** to restore the demo config."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Hyperparameters")
            model_name = gr.Dropdown(
                choices=MODEL_CHOICES,
                value=DEFAULTS["model_name"],
                label="Model",
                allow_custom_value=True,
                info="HF model ID. Use a small Qwen for fast demos on T4.",
            )
            dataset_size = gr.Slider(
                minimum=50, maximum=800, step=10, value=DEFAULTS["dataset_size"],
                label="Dataset size",
                info="Number of math problems sampled across tiers 1-10.",
            )
            max_steps = gr.Slider(
                minimum=10, maximum=300, step=5, value=DEFAULTS["max_steps"],
                label="Max GRPO steps",
                info="More steps = stronger learning but longer runtime.",
            )
            learning_rate = gr.Number(
                value=DEFAULTS["learning_rate"], label="Learning rate", precision=8,
                info="Typical range 1e-6 to 5e-5 for LoRA on small Qwen.",
            )
            batch_size = gr.Slider(
                minimum=1, maximum=4, step=1, value=DEFAULTS["batch_size"],
                label="Per-device batch size",
            )
            grad_accum = gr.Slider(
                minimum=1, maximum=8, step=1, value=DEFAULTS["grad_accum"],
                label="Gradient accumulation steps",
                info="Effective batch = batch_size * grad_accum.",
            )
            num_generations = gr.Slider(
                minimum=2, maximum=8, step=1, value=DEFAULTS["num_generations"],
                label="GRPO generations per prompt",
                info="Higher = denser advantage signal, but slower.",
            )
            max_completion = gr.Slider(
                minimum=64, maximum=512, step=32, value=DEFAULTS["max_completion"],
                label="Max completion length (tokens)",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=1.5, step=0.05, value=DEFAULTS["temperature"],
                label="Sampling temperature (training rollouts)",
            )
            lora_r = gr.Slider(
                minimum=8, maximum=64, step=8, value=DEFAULTS["lora_r"],
                label="LoRA rank (r)",
            )
            lora_alpha = gr.Slider(
                minimum=16, maximum=128, step=8, value=DEFAULTS["lora_alpha"],
                label="LoRA alpha",
                info="Common rule of thumb: alpha = 2 * r.",
            )
            warmup_ratio = gr.Slider(
                minimum=0.0, maximum=0.2, step=0.01, value=DEFAULTS["warmup_ratio"],
                label="Warmup ratio",
            )

            with gr.Row():
                start_btn = gr.Button("Start Training", variant="primary")
                reset_btn = gr.Button("Reset to Defaults")

        with gr.Column(scale=2):
            gr.Markdown("### Status & Logs")
            logs = gr.Textbox(
                label="Status + tail of train.log",
                value=get_logs(),
                lines=32,
            )
            refresh_btn = gr.Button("Refresh Logs")

    inputs = [
        model_name, dataset_size, max_steps, learning_rate,
        batch_size, grad_accum, num_generations, max_completion,
        temperature, lora_r, lora_alpha, warmup_ratio,
    ]
    start_btn.click(fn=start_training, inputs=inputs, outputs=logs)
    refresh_btn.click(fn=get_logs, inputs=None, outputs=logs)
    reset_btn.click(fn=reset_defaults, inputs=None, outputs=inputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
