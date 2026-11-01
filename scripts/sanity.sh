#!/usr/bin/env bash
# Harness check: CWEval's reference (secure) solutions must pass their own tests in every
# language. Run after any change to the container setup. Prints pass/fail counts per language.
set -euo pipefail
docker run --rm --network none --platform linux/amd64 co1lin/cweval zsh -c '
set -e
. /home/ubuntu/miniforge3/etc/profile.d/conda.sh
cd /home/ubuntu/CWEval; source .env; export GOTOOLCHAIN=local
python cweval/commons.py compile_all_in --path benchmark/ > /tmp/compile.log 2>&1 || { echo "COMPILE FAILED"; tail -5 /tmp/compile.log; }
for d in core/c core/cpp core/go core/js core/py lang/c; do
  printf "%-9s " $d; timeout 600 python -m pytest benchmark/$d -k "not _unsafe" -p no:cacheprovider -q --tb=no -n 4 2>&1 | tail -1
done'
