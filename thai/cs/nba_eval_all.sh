#!/usr/bin/env bash
# Next-best-action eval for all three businesses: eval files from the hand labels, then run 8 asked directly, the teacher
# (with context / message + playbook) and the rule (run 8 intent -> table -> context rule). Runs beside :8011 on GPU 1.
#   nohup bash thai/cs/nba_eval_all.sh > thai/out/nba_eval_all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
for biz in banking telecom insurance; do
  echo "== [$(date +%H:%M)] $biz"
  docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train python label_nba.py --business $biz --eval-only
  docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train \
    python eval_nba.py --business $biz --models /work/thai/out/laya-th-run8 --rule-model /work/thai/out/laya-th-run8 \
    --teacher http://172.18.72.145:8010 2>&1 | grep -v -iE 'warn|Fetching'
done
echo "== [$(date +%H:%M)] done"
