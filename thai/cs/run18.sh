#!/usr/bin/env bash
# Run 18, unattended: the Thai train split of hagsmand1/laya-thai-decisions (varied phrasings, object states, negated and
# relabelled yes/no, scales in both directions) on top of run 16, with a replay of the run 16 items.
# Items -> a pass from run 16 -> evals (their suites and ours) -> restore :8011 (run 16).
#   nohup bash thai/cs/run18.sh > thai/out/run18.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
mkdir -p thai/out
docker stop laya-cascade >/dev/null 2>&1 || true
docker rm -f vllm-plan >/dev/null 2>&1 || true

echo "== [$(date +%H:%M)] 1/4 items"
"${RUN[@]}" python label_cs18.py 2>&1 | grep -v -iE 'warn|Fetching|Generating|Downloading'

echo "== [$(date +%H:%M)] 2/4 train run 18 from run 16, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run16 --items /work/thai/data/cs/cs18_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run18

echo "== [$(date +%H:%M)] 3/4 evals"
"${RUN[@]}" python ../eval_decisions.py --models /work/thai/out/laya-th-run18 --out /work/thai/out/decisions18.json 2>&1 | grep -v -iE 'warn|Fetching|Generating|Downloading'
"${RUN[@]}" python eval_stars.py --models /work/thai/out/laya-th-run16,/work/thai/out/laya-th-run18 --n 3000 --out /work/thai/out/stars18.json 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run18 --out /work/thai/out/real18.json
"${RUN[@]}" python eval_play.py --models /work/thai/out/laya-th-run18 --out /work/thai/out/play18.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run18 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run18_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run18 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run18.json

echo "== [$(date +%H:%M)] 4/4 restore :8011 (run 16) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
