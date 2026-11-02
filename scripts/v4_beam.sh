#!/usr/bin/env bash
# CyberNemo v4: teacher distillation on HARD synthetic tasks only (tasks where Nemotron 120B never produced a
# passing answer). Reflection Beam answers them (its default reasoning mode, final code only), answers are
# filtered by each task's own tests in offline W&B Sandboxes, merged with v3's data, checked, and trained on
# Fireworks. Training tasks are synthetic; CWEval is never used.
set -uo pipefail
cd "$(dirname "$0")/.."
ART=.venv-art/bin/python; PY=.venv/bin/python
SD="datagen/self_distill.py --model reflection/Beam-1.0-501B-A23B --tag beam --default_mode --max_tokens 32768 --k 2 --ids_file data/synth/beam_hard_ids.txt"
echo "== $(date +%H:%M) Beam answers"
$ART $SD gen --concurrency 16 | grep -vE "^  [0-9]+/|failed:"
$ART $SD gen --concurrency 8 | grep -vE "^  [0-9]+/|failed:"
echo "== $(date +%H:%M) sandbox filter"
$ART $SD score 2>&1 | grep -vE "^I1101|fork|cwsandbox"
$ART $SD score 2>&1 | grep -vE "^I1101|fork|cwsandbox"
$ART $SD build --per_task 2 --out data/selfdistill_beam_hard.jsonl
cat data/selfdistill_v3_train.jsonl data/selfdistill_beam_hard.jsonl | awk '!seen[$0]++' > data/selfdistill_v4_train.jsonl
$ART datagen/check_sft.py data/selfdistill_v4_train.jsonl --count_tokens 40 || { echo "CHECK FAILED, not training"; exit 1; }
echo "== $(date +%H:%M) Fireworks dataset + SFT"
$PY scripts/fireworks_sft.py dataset secfix-120b-v4 data/selfdistill_v4_train.jsonl | tail -1
$PY scripts/fireworks_sft.py train secfix-120b-v4 --epochs 1 --lora_rank 16
echo "== $(date +%H:%M) submitted"
