#!/usr/bin/env bash
# CyberNemo v3: targeted data for the CWE families where v1/v2 fail on CWEval (error analysis, disclosed in README):
# CWE-113 header injection, CWE-79 XSS, CWE-918 SSRF (py/js/c/cpp), CWE-347 JWT (c/cpp), CWE-327/329 crypto (cpp).
# New synthetic tasks (teacher + sandbox verification + 1 repair round) -> 120B thinking self-distill on those
# tasks only -> merged with v1's data -> checked -> imported to Fireworks via GCS -> LoRA SFT.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python; ART=.venv-art/bin/python; S=data/synth
batch() {  # name n seed lang cwes
  $PY datagen/teacher.py --lang "$4" --cwes "$5" --n "$2" --seed "$3" --threads 4 --out $S/$1.jsonl | tail -1
  datagen/verify.sh $S/$1.jsonl
  $PY datagen/teacher.py --repair $S/$1.jsonl --threads 4 --out $S/$1.r1.jsonl | tail -1
  datagen/verify.sh $S/$1.r1.jsonl
  echo "$1 kept: $(cat $S/$1.verify.jsonl $S/$1.r1.verify.jsonl | grep -c '"ok": true')"
}
echo "== $(date +%H:%M) teacher + verify"
batch v3_py 75 901 py 113,79,918 > $S/v3_py.log 2>&1 &
batch v3_js 75 902 js 113,79,918 > $S/v3_js.log 2>&1 &
batch v3_c 100 903 c 113,79,918,347 > $S/v3_c.log 2>&1 &
batch v3_cpp 125 904 cpp 79,918,347,327,329 > $S/v3_cpp.log 2>&1 &
wait
grep -h "kept:" $S/v3_*.log
echo "== $(date +%H:%M) 120B thinking self-distill on the new tasks"
SD="datagen/self_distill.py --model nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B --tag 120b_think --thinking --max_tokens 16384 --k 4 --only _901_,_902_,_903_,_904_"
$ART $SD gen --concurrency 12 | grep -vE "^  [0-9]+/|failed:"
$ART $SD gen --concurrency 6 | grep -vE "^  [0-9]+/|failed:"
$ART $SD score 2>&1 | grep -vE "^I1101|fork|cwsandbox"
$ART $SD score 2>&1 | grep -vE "^I1101|fork|cwsandbox"
$ART $SD build --per_task 3 --out data/selfdistill_v3_targeted.jsonl
echo "== $(date +%H:%M) merge with v1 data and check"
cat data/selfdistill_120b_think_train.jsonl data/selfdistill_v3_targeted.jsonl | awk '!seen[$0]++' > data/selfdistill_v3_train.jsonl
$ART datagen/check_sft.py data/selfdistill_v3_train.jsonl --count_tokens 40 || { echo "CHECK FAILED, not training"; exit 1; }
echo "== $(date +%H:%M) Fireworks dataset + SFT"
$PY scripts/fireworks_sft.py dataset secfix-120b-v3 data/selfdistill_v3_train.jsonl | tail -1
$PY scripts/fireworks_sft.py train secfix-120b-v3 --epochs 1 --lora_rank 16
echo "== $(date +%H:%M) submitted"
