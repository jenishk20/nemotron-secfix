#!/usr/bin/env bash
# One batch end to end: generate -> verify -> (repair -> verify) x2.
# Usage: datagen/run_batch.sh <name> <n_tasks> <seed>    e.g. datagen/run_batch.sh py_b2 600 2
# Resumable: re-running skips work already in data/synth/<name>*.jsonl.
set -euo pipefail
NAME="$1"; N="$2"; SEED="$3"; THREADS="${THREADS:-12}"
cd "$(dirname "$0")/.."
PY=.venv/bin/python; S=data/synth
$PY datagen/teacher.py --n "$N" --seed "$SEED" --threads "$THREADS" --out $S/$NAME.jsonl | tail -1
datagen/verify.sh $S/$NAME.jsonl
$PY datagen/teacher.py --repair $S/$NAME.jsonl --threads "$THREADS" --out $S/$NAME.r1.jsonl | tail -1
datagen/verify.sh $S/$NAME.r1.jsonl
$PY datagen/teacher.py --repair $S/$NAME.r1.jsonl --threads "$THREADS" --out $S/$NAME.r2.jsonl | tail -1
datagen/verify.sh $S/$NAME.r2.jsonl
cat $S/$NAME.verify.jsonl $S/$NAME.r1.verify.jsonl $S/$NAME.r2.verify.jsonl | grep -c '"ok": true' | xargs echo "$NAME kept:"
