#!/usr/bin/env bash
# Overnight: the 120B recipe on Lightning 30B, all on W&B.
# 1) thinking-mode self-distill (k=1; Lightning thinks 4-15k tokens per answer) verified in W&B Sandboxes, 2) SFT on W&B Serverless Training,
# 3) CWEval at 10 answers per task for the fine-tune, scored in W&B Sandboxes (base: lightning_nothink_n10).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-art/bin/python; L=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B
SD="datagen/self_distill.py --model $L --tag lightning_think --thinking --max_tokens 16384 --k 1"
echo "== $(date +%H:%M) distill: sample"
$PY $SD gen --concurrency 12 | grep -vE "^  [0-9]+/|failed:" || true
$PY $SD gen --concurrency 6 | grep -vE "^  [0-9]+/|failed:" || true
echo "== $(date +%H:%M) distill: score"
$PY $SD score 2>&1 | grep -vE "^I1101|fork|cwsandbox" || true
$PY $SD score 2>&1 | grep -vE "^I1101|fork|cwsandbox" || true
$PY $SD build --per_task 1
$PY datagen/check_sft.py data/selfdistill_lightning_think_train.jsonl   # stops here on any problem
echo "== $(date +%H:%M) SFT on W&B"
$PY train/wandb_sft.py --name secfix-sft-v4 --run_name sft-v4 --data data/selfdistill_lightning_think_train.jsonl \
    --epochs 1 --lr 1e-4 2>&1 | tee outputs/sft-v4.train.log | grep -E "^done:|Error|error" || true
INF=$(grep -oE "inference model: wandb-artifact:///[^ ]+" outputs/sft-v4.train.log | tail -1 | sed 's/inference model: //')
[ -n "$INF" ] || { echo "SFT did not report an inference model"; exit 1; }
echo "== $(date +%H:%M) eval $INF at n=10"
$PY scripts/fast_gen.py sftv4_think_n10 "wbtrain:$INF" --n 10 --concurrency 12
$PY scripts/score_sandbox.py sftv4_think_n10 > outputs/sftv4_think_n10.sbx.log 2>&1
echo "== $(date +%H:%M) done"
