#!/usr/bin/env bash
# Run 17, unattended: run 16 with the neutral side weighted more (run 16 got "neutral" back only half-way): 6,000 Wisesight
# neutral x2 and question x3 instead of 3,000 x1 and x2, everything else as run 16, again from run 14.
# Items -> a pass from run 14 -> evals -> restore :8011 (run 16 since 2026-10-03).
#   nohup bash thai/cs/run17.sh > thai/out/run17.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
mkdir -p thai/out
docker stop laya-cascade >/dev/null 2>&1 || true
docker rm -f vllm-plan >/dev/null 2>&1 || true

echo "== [$(date +%H:%M)] 1/4 items"
"${RUN[@]}" python label_cs15.py --wisesight /work/thai/data/cc/wisesight_train.jsonl --ws-neutral 6000 --ws-neutral-repeat 2 --ws-question-repeat 3 --out /work/thai/data/cs/cs17_items.pt

echo "== [$(date +%H:%M)] 2/4 train run 17 from run 14, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run14 --items /work/thai/data/cs/cs17_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run17

echo "== [$(date +%H:%M)] 3/4 evals"
"${RUN[@]}" python eval_stars.py --models /work/thai/out/laya-th-run16,/work/thai/out/laya-th-run17 --n 3000 --out /work/thai/out/stars17.json 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run17 --out /work/thai/out/real17.json
"${RUN[@]}" python eval_play.py --models /work/thai/out/laya-th-run17 --out /work/thai/out/play17.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run17 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run17_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run17 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run17.json

echo "== [$(date +%H:%M)] 4/4 restore :8011 (run 16) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
