#!/usr/bin/env bash
# Score one generation run in the CWEval container. Usage: scripts/score.sh <run_name>
# Reads outputs/<run>/, writes res.json per sample folder + outputs/<run>/summary.json.
set -euo pipefail
RUN="$1"; ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker run --rm --network none --platform linux/amd64 \
  -v "$ROOT/outputs":/host_out -v "$ROOT/scripts":/host_scripts:ro co1lin/cweval zsh -c "
set -e
# CWEval's own container env: activates the cweval conda env and sets the C include/lib
# paths and NODE_PATH. Without it every C/C++ task fails to compile (scores 0.0).
. /home/ubuntu/miniforge3/etc/profile.d/conda.sh
cd /home/ubuntu/CWEval; source .env
export GOTOOLCHAIN=local
PY=\$(which python)
mkdir -p evals && rm -rf evals/$RUN && cp -r /host_out/$RUN evals/
\$PY /host_scripts/score_in_container.py evals/$RUN ${SCORE_ARGS:-}
for d in evals/$RUN/generated_*; do cp \$d/res.json /host_out/$RUN/\$(basename \$d)/; done
cp evals/$RUN/summary.json /host_out/$RUN/
"
