#!/bin/bash
# Serve a model with vLLM and generate CWEval answers (all 119 tasks, n=5, temp 0.8, thinking off).
# Scoring happens on the laptop in Docker (copy outputs/<run> back).
#   MODEL=models/lightning-bf16 RUN=lightning_base_n5 sbatch cluster/eval.sbatch
#   MODEL=runs/lora_v1/merged  RUN=lightning_lora_v1_n5 sbatch cluster/eval.sbatch
#   THINK=on to evaluate with reasoning enabled; VLLM_ENV=/scratch/$USER/<venv> to use an existing venv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:1
#SBATCH --time=03:00:00
#SBATCH --job-name=secfix-eval
#SBATCH --mem=128GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/scratch/%u/nemotron-secfix/logs/eval_%j.out
#SBATCH --error=/scratch/%u/nemotron-secfix/logs/eval_%j.err
set -eo pipefail
cd /scratch/$USER/nemotron-secfix
: "${MODEL:?set MODEL}" "${RUN:?set RUN}"
VENV="$(cd "${VLLM_ENV:-env_vllm}" && pwd)"
export HF_HOME=/scratch/$USER/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
"$VENV"/bin/vllm serve "$MODEL" --served-model-name secfix --port 8000 --max-model-len 16384 \
  --max-num-seqs 128 --enable-prefix-caching --mamba-ssm-cache-dtype float16 \
  --reasoning-parser nemotron_v3 > logs/vllm_$RUN.log 2>&1 &
VLLM=$!
trap 'kill $VLLM 2>/dev/null || true' EXIT
for i in $(seq 1 120); do curl -sf localhost:8000/health >/dev/null && break; sleep 10; done
curl -sf localhost:8000/health >/dev/null || { echo "vLLM did not start"; tail -50 logs/vllm_$RUN.log; exit 1; }

EXTRA=()
[[ "${THINK:-off}" == off ]] && EXTRA=(--extra_body "{'chat_template_kwargs': {'enable_thinking': False}}")
OVERWRITE=n; [[ -d outputs/$RUN ]] && OVERWRITE=y   # resume
cd third_party/CWEval
echo "$OVERWRITE" | OPENAI_API_KEY=EMPTY PYTHONPATH=$PWD "$VENV"/bin/python ../../scripts/gen_threaded.py \
  --threads 32 --model openai/secfix --api_base http://localhost:8000/v1 \
  --n 5 --temperature 0.8 --max_completion_tokens 8192 --eval_path ../../outputs/$RUN "${EXTRA[@]}"
cd ../..
echo "$RUN: $(find outputs/$RUN -name '*_raw.*' | wc -l) answers, $(find outputs/$RUN -name '*_raw.*' -empty | wc -l) empty"
