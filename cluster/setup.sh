#!/usr/bin/env bash
# One-time setup on the cluster (run on a node WITH internet, e.g. the login node).
# Clones CWEval (pinned + our patch), creates two venvs (training, vLLM serving),
# downloads the model (~66 GB). Safe to re-run.
#   cd /scratch/$USER/nemotron-secfix && bash cluster/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export HF_HOME=/scratch/$USER/hf_cache PIP_CACHE_DIR=/scratch/$USER/pip_cache
mkdir -p logs outputs runs models

# Same CWEval commit as the laptop, plus the one-line fix: write '' when the model returns no answer
if [[ ! -d third_party/CWEval ]]; then
  git clone https://github.com/Co1lin/CWEval third_party/CWEval
  git -C third_party/CWEval checkout e9a2a124c8c53679b6d8d27adfd2f6c40e7576d7
  sed -i "s/                f.write(resp)$/                f.write(resp or '')  # None = model returned no answer; score as a failure/" \
    third_party/CWEval/cweval/generate.py
fi
grep -q "f.write(resp or '')" third_party/CWEval/cweval/generate.py && echo "CWEval patched"

if [[ ! -d env_train ]]; then
  $PY -m venv env_train
  env_train/bin/pip install -U pip
  env_train/bin/pip install torch "transformers==4.57.6" peft accelerate datasets
fi
if [[ ! -d env_vllm ]]; then
  $PY -m venv env_vllm
  env_vllm/bin/pip install -U pip
  env_vllm/bin/pip install "vllm==0.27.1" litellm fire natsort p_tqdm python-dotenv
fi
env_train/bin/python -c "import torch, transformers, peft; print('train env: torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
env_vllm/bin/python -c "import vllm, litellm; print('vllm env: vllm', vllm.__version__)"

env_train/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
                      local_dir="models/lightning-bf16", allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE"])
print("model at", p)
PY
du -sh models/lightning-bf16
