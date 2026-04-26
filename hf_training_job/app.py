import subprocess
import threading
from pathlib import Path

import gradio as gr

LOG_FILE = Path("train.log")
STATUS_FILE = Path("status.txt")
RUN_LOCK = threading.Lock()
TRAINING_THREAD: threading.Thread | None = None
APP_VERSION = "restart-v1-clear-stale-logs"


def _set_status(text: str) -> None:
    STATUS_FILE.write_text(text, encoding="utf-8")


def _read_tail(max_lines: int = 120) -> str:
    if not LOG_FILE.exists():
        return "No logs yet."
    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    tail = lines[-max_lines:]
    return "\n".join(tail) if tail else "No logs yet."


def _clear_run_files() -> None:
    LOG_FILE.write_text("", encoding="utf-8")
    _set_status("starting")


def run_training() -> None:
    _set_status("running")
    with LOG_FILE.open("w", encoding="utf-8") as out:
        out.write(f"=== Starting background training process ({APP_VERSION}) ===\n")
        out.flush()
        try:
            subprocess.run(
                ["python", "train_worker.py"],
                check=True,
                stdout=out,
                stderr=subprocess.STDOUT,
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


def start_training() -> str:
    global TRAINING_THREAD
    with RUN_LOCK:
        if TRAINING_THREAD and TRAINING_THREAD.is_alive():
            return "Training is already running.\n\n" + get_logs()
        _clear_run_files()
        TRAINING_THREAD = threading.Thread(target=run_training, daemon=True)
        TRAINING_THREAD.start()
    return get_logs()


# Start training automatically on app startup, after clearing stale logs.
start_training()


def get_logs() -> str:
    status = STATUS_FILE.read_text(encoding="utf-8").strip() if STATUS_FILE.exists() else "starting"
    return f"App version: {APP_VERSION}\nStatus: {status}\n\n{_read_tail()}"


with gr.Blocks() as demo:
    gr.Markdown("# Math Escalation GRPO Training")
    gr.Markdown("Training starts automatically in the background.")
    gr.Markdown(
        "If this Space fails, click **Refresh Status** and share the tail logs. "
        "Most setup/runtime issues are visible there."
    )

    logs = gr.Textbox(label="Status + Logs", value=get_logs(), lines=24)
    refresh_btn = gr.Button("Refresh Status")
    restart_btn = gr.Button("Restart Training")
    refresh_btn.click(fn=get_logs, outputs=logs)
    restart_btn.click(fn=start_training, outputs=logs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
