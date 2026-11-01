"""Central config for the secure-code-fix hackathon build.

Hero model: NVIDIA Nemotron 3.5 Super VL 120B (BF16) on Fireworks.
Why this one: it's the only Nemotron on Fireworks that supports the full
tuning range (LoRA SFT + DPO + GRPO RL) on the serverless training pool.

NOTE: every model path / price below should be re-checked against the live
Fireworks dashboard this weekend. Prices move and the serverless-training
catalog is generated dynamically.
"""

# --- Fireworks model paths ---------------------------------------------------
# Hero base (supports SFT + DPO + GRPO RL on serverless training):
NEMOTRON_VL_120B = "accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16"
# Alternative non-VL 120B (LoRA SFT + on-demand; confirm RL support before using):
NEMOTRON_SUPER_120B = "accounts/fireworks/models/nemotron-3-super-120b-a12b-bf16"

# The base we actually train/eval. Keep ONE base across SFT + RL so the
# before/after comparison is clean.
BASE_MODEL = NEMOTRON_VL_120B

# Output model ids (must be lowercase, <=17 chars for checkpoint names).
SFT_OUTPUT_MODEL = "nemo-secfix-sft"
RL_OUTPUT_MODEL = "nemo-secfix-rl"

# --- Frontier baseline -------------------------------------------------------
# CWEval public leaderboard (cite directly if you don't re-run it):
#   gpt-4o-2024-08-06      func@10 90.71   func-sec@10 65.33
#   claude-3-5-sonnet      func@10 91.36   func-sec@10 59.20
#   Llama 3.1 8B Instruct  func@10 80.54   func-sec@10 48.70
# If you want a live number, run OpenAI gpt-4o through the same harness.
FRONTIER_BASELINE_FUNC_SEC = 65.33  # gpt-4o-2024-08-06, for the slide

# --- Eval --------------------------------------------------------------------
CWEVAL_REPO = "https://github.com/Co1lin/CWEval"
CWEVAL_DOCKER_IMAGE = "co1lin/cweval"   # prebuilt; isolates untrusted generated code
# Iterate on a subset to save money/time; score the full set only for the final table.
CWEVAL_SUBSET_SIZE = 60    # ~half of the ~119 tasks
CWEVAL_SAMPLES_PER_TASK = 5  # pass@1 with a few samples; bump for the final run

# --- Training data -----------------------------------------------------------
CVEFIXES_SQLITE = "data/CVEfixes.db"      # downloaded from Zenodo (SQL dump -> sqlite)
TRAIN_JSONL = "data/train_secfix.jsonl"   # chat JSONL for Fireworks SFT
# Languages to keep (match whatever CWEval covers so the task distributions align):
TRAIN_LANGUAGES = ["Python", "C", "JavaScript", "Go"]

# --- Weights & Biases --------------------------------------------------------
WANDB_PROJECT = "nemotron-secfix"
WANDB_ENTITY = None  # set to your TEAM entity (W&B Inference needs a Team, not personal)

# --- Rough cost budget (per day; re-verify) ----------------------------------
# Serverless training (VL 120B, 50% promo): prefill $0.57 / sample $1.44 / train $2.32 per 1M tok
# On-demand serving (120B BF16, ~4 GPUs): ~$32/hr (4x H100) to ~$52/hr (4x B200)
# SFT run ~$15-20 | small GRPO run ~$30-60 | serving 4h ~$130 => plan for ~$200-400.

# --- W&B Inference (cheap base-model baseline) -------------------------------
# OpenAI-compatible. NVFP4-quantized, so it is NOT the same weights we tune on
# Fireworks (BF16): use it for the quick baseline, re-run the official base row
# on Fireworks before comparing against the tuned model.
WANDB_INFERENCE_BASE_URL = "https://api.inference.wandb.ai/v1"
WANDB_NEMOTRON = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B"
