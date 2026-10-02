#!/usr/bin/env bash
# Run 16, unattended: run 15 again with neutral / question examples. Run 15 fixed sarcasm but stopped answering "neutral"
# (every new sentiment item was positive or negative), so the same items plus the Wisesight training split, again from run 14.
# Items -> a pass from run 14 -> evals -> restore :8011 (still run 14).
#   nohup bash thai/cs/run16.sh > thai/out/run16.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
mkdir -p thai/out
docker stop laya-cascade >/dev/null 2>&1 || true
docker rm -f vllm-plan >/dev/null 2>&1 || true

echo "== [$(date +%H:%M)] 1/4 items"
"${RUN[@]}" python label_cs15.py --wisesight /work/thai/data/cc/wisesight_train.jsonl --out /work/thai/data/cs/cs16_items.pt

echo "== [$(date +%H:%M)] 2/4 train run 16 from run 14, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run14 --items /work/thai/data/cs/cs16_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run16

echo "== [$(date +%H:%M)] 3/4 evals"
"${RUN[@]}" python eval_stars.py --models /work/thai/out/laya-th-run14,/work/thai/out/laya-th-run16 --n 3000 --out /work/thai/out/stars16.json 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run16 --out /work/thai/out/real16.json
"${RUN[@]}" python eval_play.py --models /work/thai/out/laya-th-run16 --out /work/thai/out/play16.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run16 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run16_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run16 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run16.json

echo "== [$(date +%H:%M)] 4/4 restore :8011 (run 14) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
