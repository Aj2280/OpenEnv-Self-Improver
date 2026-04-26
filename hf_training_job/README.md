---
title: Math Escalation Trainer
sdk: docker
app_port: 7860
pinned: false
---

# Math Escalation Training Job

Gradio app: pick **5** training options, then **Start Training**. After GRPO, weights are pushed to the Hub when allowed.

## One-time Hub setup (fixes `403` on `create`)

Some Space tokens can **upload** to an existing model repo but get **403** when the code calls the **create repo** API.

1. Run training once and read the log line: `[train_worker] Hub output repo: YOUR_USER/math-escalation-grpo-...`
2. Open **[Create new model](https://huggingface.co/new-model)** with the **same owner** and **same repo name** (empty repo is fine).
3. Run **Start Training** again — the worker skips `create_repo` and pushes to that repo.

Optional Space secrets:

- **`HF_TOKEN`** — personal access token with **Write** (if you prefer not to pre-create repos).
- **`HF_OUTPUT_REPO`** — override push target, e.g. `Abhi2280/my-custom-model-name`.
