#!/usr/bin/env bash
# The public checkpoint, run 14 and run 16 (served) on the eval suites of hagsmand1/laya-thai-decisions. Runs beside the
# cascade on GPU 1; nothing is stopped.
#   nohup bash thai/run_decisions.sh > thai/out/decisions.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${1:-convaiinnovations/laya-multilingual,/work/thai/out/laya-th-run14,/work/thai/out/laya-th-run16}
OUT=${2:-/work/thai/out/decisions.json}
docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf --shm-size 2g laya-train \
  python eval_decisions.py --models "$MODELS" --out "$OUT" 2>&1 | grep -v -iE 'warn|Fetching|Generating|Downloading'
docker run --rm -v "$PWD":/work laya-train chmod -R a+rX /work/thai/out
echo "== done"
