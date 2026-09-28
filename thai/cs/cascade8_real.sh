#!/usr/bin/env bash
# Cascade threshold sweep of run 8 on the real Pantip set (student on GPU 1, teacher :8010); no option-count gate.
#   nohup bash thai/cs/cascade8_real.sh > thai/out/cascade8_real.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf laya-train \
  python cascade.py --student /work/thai/out/laya-th-run8 --eval /work/thai/data/domain/real_cs_eval.jsonl --max-options 0 \
  --out /work/thai/out/cascade8_real.json
echo "== [$(date +%H:%M)] done"
