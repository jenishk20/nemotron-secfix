#!/usr/bin/env bash
# Generate CWEval answers for one model. Usage:
#   scripts/gen.sh <run_name> <model> [extra generate.py args...]
# <model> forms:
#   wandb:<id>        W&B Inference, e.g. wandb:nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B
#   openrouter/<id>   OpenRouter via litellm, e.g. openrouter/anthropic/claude-sonnet-5.5
#                     (set GEN_THREADS=1: CWEval's forked workers hang on OpenRouter on macOS)
#   wbtrain:<id>      a model fine-tuned on W&B Serverless Training (id from train/wandb_sft.py's
#                     "inference model:" line); base URL from WBTRAIN_BASE_URL (its "base url:")
#   fireworks:<id>    a Fireworks model or deployment, e.g. fireworks:accounts/<acct>/models/<lora>
#                     (uses FIREWORKS_API_KEY; LoRAs need a running dedicated deployment)
#   vertex_ai/<id>    Gemini on Vertex AI, e.g. vertex_ai/gemini-3.5-flash (also needs GEN_THREADS=1);
#                     uses GCP_PROJECT, GCP_LOCATION and GOOGLE_ADC_B64 from .env
# Defaults: n=5, temperature 0.8, 8192 max tokens; pass e.g. --langs '["py"]' --ppt secure to override.
# Writes outputs/<run_name>/ and outputs/<run_name>.gen.log. Resumable (re-run to fill gaps).
set -euo pipefail
RUN="$1"; MODEL="$2"; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
set -a; source "$ROOT/.env"; set +a
cd "$ROOT/third_party/CWEval"; export PYTHONPATH="$PWD"
ARGS=(--n 5 --temperature 0.8 --max_completion_tokens 8192 --num_proc 4 --eval_path "$ROOT/outputs/$RUN")
if [[ "$MODEL" == wandb:* ]]; then
  export OPENAI_API_KEY="$WANDB_API_KEY"
  ARGS+=(--model "openai/${MODEL#wandb:}" --api_base https://api.inference.wandb.ai/v1)
elif [[ "$MODEL" == wbtrain:* ]]; then
  : "${WBTRAIN_BASE_URL:?set WBTRAIN_BASE_URL to the base url printed by train/wandb_sft.py}"
  export OPENAI_API_KEY="$WANDB_API_KEY"
  ARGS+=(--model "openai/${MODEL#wbtrain:}" --api_base "$WBTRAIN_BASE_URL")
elif [[ "$MODEL" == fireworks:* ]]; then
  export OPENAI_API_KEY="$FIREWORKS_API_KEY"
  ARGS+=(--model "openai/${MODEL#fireworks:}" --api_base https://api.fireworks.ai/inference/v1)
else
  ARGS+=(--model "$MODEL")
fi
if [[ "$MODEL" == vertex_ai/* ]]; then
  # Decode the credentials to a private temp file for this run only
  ADC="$(mktemp)"; chmod 600 "$ADC"; trap 'rm -f "$ADC"' EXIT
  echo "$GOOGLE_ADC_B64" | base64 -d > "$ADC"
  export GOOGLE_APPLICATION_CREDENTIALS="$ADC" VERTEXAI_PROJECT="$GCP_PROJECT" VERTEXAI_LOCATION="$GCP_LOCATION"
fi
OVERWRITE=n; [[ -d "$ROOT/outputs/$RUN" ]] && OVERWRITE=y   # resume into an existing run
# GEN_THREADS=1: thread-based generator (needed for OpenRouter on macOS; see gen_threaded.py)
if [[ "${GEN_THREADS:-0}" == 1 ]]; then CMD=("$ROOT/scripts/gen_threaded.py"); else CMD=(cweval/generate.py gen); fi
echo "$OVERWRITE" | "$ROOT/.venv/bin/python" "${CMD[@]}" "${ARGS[@]}" "$@" \
  > "$ROOT/outputs/$RUN.gen.log" 2>&1
echo "$RUN: $(find "$ROOT/outputs/$RUN" -name '*_raw.*' | wc -l | tr -d ' ') answers," \
     "$(find "$ROOT/outputs/$RUN" -name '*_raw.*' -empty | wc -l | tr -d ' ') empty"
