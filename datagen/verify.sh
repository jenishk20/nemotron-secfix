#!/usr/bin/env bash
# Verify teacher candidates in the offline CWEval container.
# Usage: datagen/verify.sh data/synth/<name>.jsonl   -> writes data/synth/<name>.verify.jsonl
set -euo pipefail
IN="$1"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="$(basename "$IN" .jsonl)"
docker run --rm --network none --platform linux/amd64 \
  -v "$ROOT/data/synth":/synth -v "$ROOT/datagen":/datagen:ro co1lin/cweval zsh -c "
set -e
. /home/ubuntu/miniforge3/etc/profile.d/conda.sh
cd /home/ubuntu/CWEval; source .env
python /datagen/verify_in_container.py /synth/$NAME.jsonl /synth/$NAME.verify.jsonl ${VERIFY_ARGS:-}
"
