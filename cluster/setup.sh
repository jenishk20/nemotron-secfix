#!/usr/bin/env bash
# One-time setup on the cluster (run on a node WITH internet, e.g. the login node).
# Clones CWEval (pinned + our patch), creates two venvs (training, vLLM serving),
# downloads the model (~66 GB). Safe to re-run.
#   cd /scratch/$USER/nemotron-secfix && bash cluster/setup.sh
# To reuse venvs you already have, point at them (nothing is installed into them):
#   TRAIN_ENV=/scratch/$USER/glassbox_env VLLM_ENV=/scratch/$USER/passk_env bash cluster/setup.sh
# If vLLM comes from the official container (VLLM_SIF=/path/vllm.sif), the env only gets the
# small CWEval client packages, not vLLM itself. SKIP_VLLM=1 sets up training only.
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

TRAIN_ENV="${TRAIN_ENV:-env_train}"; VLLM_ENV="${VLLM_ENV:-env_vllm}"
if [[ ! -d "$TRAIN_ENV" ]]; then
  $PY -m venv "$TRAIN_ENV"
  "$TRAIN_ENV"/bin/pip install -U pip
  # CUDA 12.8 build: matches the cluster driver (570.x, CUDA 12.8)
  "$TRAIN_ENV"/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
  "$TRAIN_ENV"/bin/pip install "transformers==5.22.1" peft accelerate datasets
fi
if [[ -z "${SKIP_VLLM:-}" && ! -d "$VLLM_ENV" ]]; then
  $PY -m venv "$VLLM_ENV"
  "$VLLM_ENV"/bin/pip install -U pip
  if [[ -n "${VLLM_SIF:-}" ]]; then
    "$VLLM_ENV"/bin/pip install litellm fire natsort p_tqdm python-dotenv numpy tqdm
  else
    "$VLLM_ENV"/bin/pip install "vllm==0.27.1" litellm fire natsort p_tqdm python-dotenv
  fi
fi
"$TRAIN_ENV"/bin/python -c "import torch, transformers, peft; print('train env: torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__)"
if [[ -n "${SKIP_VLLM:-}" ]]; then
  echo "skipping vLLM / eval env (SKIP_VLLM=1)"
elif [[ -n "${VLLM_SIF:-}" ]]; then
  "$VLLM_ENV"/bin/python -c "import litellm, fire, natsort, p_tqdm; print('client env ok')"
  apptainer exec --nv "$VLLM_SIF" python3 -c "import vllm; print('vllm (container):', vllm.__version__)"
else
  "$VLLM_ENV"/bin/python -c "import litellm, fire, natsort, p_tqdm; print('client env ok')"
  "$VLLM_ENV"/bin/python -c "import vllm; print('vllm (venv):', vllm.__version__)"
fi

"$TRAIN_ENV"/bin/python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
                      local_dir="models/lightning-bf16", allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE"])
print("model at", p)
PY
du -sh models/lightning-bf16
