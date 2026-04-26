import gradio as gr
import threading
import subprocess
import os
from huggingface_hub import HfApi

def run_training():
    print("Starting background training process...")
    try:
        subprocess.run(["python", "train_worker.py"], check=True)
        print("Training process finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Training process failed with exit code {e.returncode}")
    except Exception as e:
        print(f"Training process encountered an error: {e}")

# Start training automatically
threading.Thread(target=run_training, daemon=True).start()

def get_logs():
    return "Training is running in the background. Check the Space logs for detailed progress."

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Math Escalation GRPO Training")
    gr.Markdown("The training script has been launched automatically in the background.")
    gr.Markdown("1. Connects to Hugging Face\n2. Generates dataset\n3. Runs GRPO training\n4. Pushes model to Hub\n5. Pauses this Space")
    
    logs = gr.Textbox(label="Status", value="Running...", lines=5)
    refresh_btn = gr.Button("Refresh Status")
    refresh_btn.click(fn=get_logs, outputs=logs)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
