import os
import sys
import random
import re
import math
import importlib.machinery
from unittest.mock import MagicMock

# Pre-populate sys.modules with mocks for optional TRL dependencies BEFORE
# any `from trl import ...`. TRL's callbacks.py and judges.py do unconditional
# `import llm_blender` and `import weave`, neither of which we use for GRPO.
#
# - llm_blender on PyPI is broken: it imports TRANSFORMERS_CACHE from
#   transformers.utils.hub which was removed in transformers>=4.38.
# - weave (W&B tracing) is heavy and not needed for offline training.
#
# importlib.util.find_spec() reads module.__spec__ and raises ValueError if
# it's None or missing. We attach a real ModuleSpec so find_spec succeeds,
# and any subsequent `import X` returns our MagicMock.
def _install_module_mock(name: str):
    mock = MagicMock()
    mock.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mock.__name__ = name
    mock.__file__ = "<mocked>"
    mock.__path__ = []
    sys.modules[name] = mock
    return mock

for _name in [
    "llm_blender",
    "llm_blender.blender",
    "llm_blender.blender.blender",
    "llm_blender.blender.blender_utils",
    "llm_blender.pair_ranker",
    "llm_blender.pair_ranker.pairrm",
    "weave",
    "weave.trace",
    "weave.trace.context",
]:
    _install_module_mock(_name)

import torch

# Unsloth must load before transformers / peft / trl so its patches apply.
_HAS_UNSLOTH = True
try:
    import unsloth  # noqa: F401
except Exception as _unsloth_import_err:
    _HAS_UNSLOTH = False
    _UNSLOTH_IMPORT_ERR = _unsloth_import_err

import datasets
from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM

print("[train_worker] reward_fn API: v3 *args/**kwargs (fix TRL answer kw)")

# 1. Login to Hugging Face
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set!")
login(token=hf_token)
api = HfApi(token=hf_token)

# Model configuration
MODEL_NAME = "unsloth/Qwen2.5-0.5B-Instruct"


def resolve_output_repo() -> str:
    configured = os.environ.get("OUTPUT_REPO") or os.environ.get("HF_OUTPUT_REPO")
    if configured:
        return configured
    try:
        username = api.whoami(token=hf_token)["name"]
        return f"{username}/Math-Escalation-GRPO-0.5B"
    except Exception as e:
        print(f"Notice: Could not resolve HF token username ({e}); using local fallback name.")
        return "Math-Escalation-GRPO-0.5B"


OUTPUT_REPO = resolve_output_repo()

# 2. Math Environment Setup
def generate_problem(difficulty: int):
    d = difficulty
    if d == 1:
        a, b = random.randint(1, 10), random.randint(1, 10)
        return f"{a} + {b}", float(a + b)
    elif d == 2:
        a, b = random.randint(10, 99), random.randint(10, 99)
        return f"{a} + {b}", float(a + b)
    elif d == 3:
        a, b = sorted([random.randint(10, 99), random.randint(10, 99)], reverse=True)
        return f"{a} - {b}", float(a - b)
    elif d == 4:
        a, b = random.randint(2, 12), random.randint(2, 12)
        return f"{a} * {b}", float(a * b)
    elif d == 5:
        a, b, c = random.randint(2, 10), random.randint(2, 10), random.randint(1, 20)
        return f"{a} * {b} + {c}", float(a * b + c)
    elif d == 6:
        a, b, c = random.randint(2, 10), random.randint(2, 10), random.randint(1, 20)
        return f"{a} * ({b} + {c})", float(a * (b + c))
    elif d == 7:
        a, b = random.randint(5, 20), random.randint(2, 9)
        return f"{a * b} / {b}", float(a)
    elif d == 8:
        a, x = random.randint(2, 9), random.randint(1, 15)
        b = random.randint(1, 30)
        c = a * x + b
        return f"Solve for x: {a}x + {b} = {c}", float(x)
    elif d == 9:
        x = random.randint(2, 20)
        return f"sqrt({x * x})", float(x)
    else:
        a, x_val, div = random.randint(2, 5), random.randint(1, 10), random.randint(2, 4)
        b = div * random.randint(2, 8) - a * x_val
        e = (a * x_val + b) // div
        return f"Solve for x: ({a}x + {b}) / {div} = {e}", float(x_val)

def check_answer(expected: float, model_answer: float) -> float:
    return 1.0 if abs(model_answer - expected) < 1e-4 else -0.5

# 3. Reward Function
# TRL GRPO calls: reward_func(prompts=..., completions=..., completion_ids=..., **reward_kwargs).
# Ground truth lives in reward_kwargs under the dataset column name (we use "answer"), not "answers".
# Use *args/**kwargs only so a stale (completions, prompts, answers) parameter order can never break calls.
def compute_reward(*args, **kwargs):
    prompts = kwargs.get("prompts")
    completions = kwargs.get("completions")
    _ = kwargs.get("completion_ids")  # unused
    answers = kwargs.get("answer") or kwargs.get("answers")
    if len(args) >= 3 and answers is None:
        answers = args[2]
    if len(args) >= 2 and (prompts is None or completions is None):
        completions, prompts = args[0], args[1]
    if prompts is None or completions is None:
        raise ValueError(
            "compute_reward: missing prompts/completions "
            f"(args={len(args)}, kwargs={sorted(kwargs.keys())})"
        )
    if answers is None:
        raise ValueError(
            "compute_reward: need dataset column 'answer' (or kw 'answers'). "
            f"kwargs keys: {sorted(k for k in kwargs if k not in ('trainer_state', 'log_extra', 'log_metric'))}"
        )
    rewards = []
    for completion, prompt, expected in zip(completions, prompts, answers):
        text = completion[0]['content'] if isinstance(completion, list) else str(completion)

        # Component 1: Reasoning format reward
        format_reward = 0.0
        if '<thought>' in text and '</thought>' in text:
            format_reward += 0.5
            m = re.search(r'<thought>(.*?)</thought>', text, re.DOTALL)
            if m and len(m.group(1).strip()) > 10:
                format_reward += 0.2  

        # Component 2: Numeric presence reward
        clean = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL)
        nums = re.findall(r'-?\d+\.?\d*', clean)
        numeric_reward = 0.3 if nums else 0.0

        # Component 3: Correctness reward
        env_reward = -0.5
        if nums:
            try:
                predicted = float(nums[-1])
                expected_val = float(expected)
                env_reward = check_answer(expected_val, predicted)
            except Exception:
                env_reward = -0.5

        total = env_reward + format_reward + numeric_reward
        rewards.append(total)

    return rewards

# 4. Dataset Building
def build_curriculum_dataset(n_problems=400):
    print(f"📊 Building curriculum dataset with {n_problems} problems...")
    data = []
    tier_counts = {
        1: 60, 2: 60, 3: 50, 4: 50, 5: 40,
        6: 40, 7: 30, 8: 30, 9: 20, 10: 20
    }
    for tier, count in tier_counts.items():
        for _ in range(count):
            problem_str, answer = generate_problem(tier)
            prompt = (
                f"Solve the following math problem. "
                f"Think step-by-step inside <thought> tags, "
                f"then give ONLY the final numeric answer.\n\n"
                f"[Tier {tier}/10] Problem: {problem_str}\n"
                f"Answer:"
            )
            data.append({'prompt': prompt, 'answer': str(answer)})

    random.shuffle(data)
    return datasets.Dataset.from_list(data)

# 5. Load Model & Train (trl after unsloth import above)
from trl import GRPOConfig, GRPOTrainer

USE_UNSLOTH = _HAS_UNSLOTH
if USE_UNSLOTH:
    from unsloth import FastLanguageModel
else:
    print(f"⚠️ Unsloth unavailable in this environment: {_UNSLOTH_IMPORT_ERR}")
    print("⚠️ Falling back to standard Transformers model loading.")

if USE_UNSLOTH:
    print("Loading Qwen2.5-0.5B-Instruct with Unsloth 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=512,
        load_in_4bit=True,
        fast_inference=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha=32,
        lora_dropout=0,
        bias='none',
        use_gradient_checkpointing='unsloth',
        random_state=3407,
    )
else:
    print("Loading Qwen2.5-0.5B-Instruct with Transformers fallback...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

train_dataset = build_curriculum_dataset(400)

# Compatibility shim: TRL 0.12 does `model.warnings_issued["estimate_tokens"] = True`.
# transformers>=5 removed `warnings_issued`; PEFT forwards missing attrs to the inner
# causal LM, so every module in the delegation chain needs a real dict in __dict__.
def _patch_warnings_issued(root):
    stack = [root]
    seen = set()
    while stack:
        obj = stack.pop()
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)
        if isinstance(obj, torch.nn.Module):
            if not isinstance(obj.__dict__.get("warnings_issued"), dict):
                obj.__dict__["warnings_issued"] = {}
        for attr in ("base_model", "model", "module"):
            try:
                inner = getattr(obj, attr, None)
            except Exception:
                inner = None
            if inner is not None and id(inner) != oid:
                stack.append(inner)
        get_bm = getattr(obj, "get_base_model", None)
        if callable(get_bm):
            try:
                inner = get_bm()
                if inner is not None and id(inner) != oid:
                    stack.append(inner)
            except Exception:
                pass
        for child in getattr(obj, "_modules", {}).values():
            if child is not None and id(child) != oid:
                stack.append(child)

_patch_warnings_issued(model)

training_args = GRPOConfig(
    output_dir="./math_grpo",
    num_train_epochs=1,
    # Hugging Face demo run: cap training at 50 steps to reduce T4 runtime.
    # Increase/remove max_steps for a longer production-quality run.
    max_steps=50,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    logging_steps=5,
    save_steps=50,
    warmup_ratio=0.05,
    lr_scheduler_type='cosine',
    report_to='none',
    num_generations=4,
    max_completion_length=128,
    temperature=0.9,
    gradient_checkpointing=True,
    use_cpu=(not torch.cuda.is_available()),
    fp16=torch.cuda.is_available(),
    bf16=False,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    reward_funcs=compute_reward,
    processing_class=tokenizer,
)

print("🔥 Starting GRPO Training...")
trainer.train()
print("✅ Training complete!")

# 6. Push to Hugging Face Hub
print(f"🚀 Pushing model to {OUTPUT_REPO}...")
push_succeeded = False
try:
    api.create_repo(repo_id=OUTPUT_REPO, exist_ok=True, private=False, token=hf_token)
    if USE_UNSLOTH:
        # Push using Unsloth's merged method so it can be loaded directly
        # with AutoModelForCausalLM.
        try:
            model.push_to_hub_merged(OUTPUT_REPO, tokenizer, save_method="merged_16bit", token=hf_token)
            print("✅ Model pushed to Hub successfully!")
            push_succeeded = True
        except Exception as e:
            print(f"Notice: merged push failed: {e}")
            print("Trying to push adapters only...")
            model.push_to_hub(OUTPUT_REPO, token=hf_token)
            tokenizer.push_to_hub(OUTPUT_REPO, token=hf_token)
            print("✅ Adapters pushed to Hub.")
            push_succeeded = True
    else:
        # Standard Transformers fallback push.
        model.push_to_hub(OUTPUT_REPO, token=hf_token)
        tokenizer.push_to_hub(OUTPUT_REPO, token=hf_token)
        print("✅ Transformers model pushed to Hub.")
        push_succeeded = True
except Exception as e:
    print(f"⚠️ Hub upload skipped: {e}")
    print(
        "Training finished, but the HF_TOKEN in this Space cannot create/upload "
        f"to '{OUTPUT_REPO}'. Use a token with write permission, or set HF_OUTPUT_REPO "
        "to a namespace owned by that token."
    )
    local_dir = "./math_grpo_final"
    print(f"Saving trained adapters locally to {local_dir} so the run can finish cleanly...")
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    print("✅ Local adapter save complete.")

# 7. Pause Space (Auto-downscale)
try:
    print("💤 Pausing Hugging Face Space to save credits...")
    # SPACE_ID is a reserved env var on HF Spaces; use HF_SPACE_ID.
    # Fall back to this trainer space id if the secret is missing.
    space_id = os.environ.get("HF_SPACE_ID", "Abhi2280/Math-Escalation-Trainer")
    if space_id:
        api.pause_space(repo_id=space_id)
        print("✅ Space paused.")
    else:
        print("Notice: SPACE_ID env var not found, skipping pause.")
except Exception as e:
    print(f"❌ Failed to pause space: {e}")

print("🎉 Job fully complete.")
