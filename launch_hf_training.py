from huggingface_hub import HfApi, create_repo
import os

token = os.getenv("HF_TOKEN")
if not token:
    raise SystemExit(
        "Missing HF_TOKEN. Export a Hugging Face access token, e.g.\n"
        "  export HF_TOKEN=hf_...\n"
        "and rerun."
    )
api = HfApi(token=token)

repo_id = "Abhi2280/Math-Escalation-Trainer"

print(f"Creating Space {repo_id}...")
create_repo(repo_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True, token=token)

print("Uploading files...")
api.upload_folder(
    folder_path="hf_training_job",
    repo_id=repo_id,
    repo_type="space"
)

print("Setting secrets...")
# The training space needs a token to push artifacts/models back to the Hub.
# We read it from your local environment and store it as a Space secret.
api.add_space_secret(repo_id, "HF_TOKEN", token)
api.add_space_secret(repo_id, "SPACE_ID", repo_id)

print("Requesting T4-small GPU hardware...")
try:
    api.request_space_hardware(repo_id, "t4-small")
    print("✅ Requested T4-small GPU successfully.")
except Exception as e:
    print(f"Notice or Error requesting hardware: {e}")
    print("If it says 'hardware is already t4-small', that is fine.")

print("Space is booting up! Training will commence automatically in the background.")
print(f"Check logs at: https://huggingface.co/spaces/{repo_id}")
