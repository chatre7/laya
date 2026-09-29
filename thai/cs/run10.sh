#!/usr/bin/env bash
# Run 10 (cheap experiment, ~2 h): a short second pass from run 8 on the hand-labelled Pantip items only (the 1,932 real rows,
# every labelled question repeated 6x + the teacher's shared questions = ~52k items), lower learning rates, 2 epochs. Tests whether
# the real rows were simply drowned by the 300k synthetic items in runs 8/9. Evals: real Pantip set, cs9 held-out, public set.
#   nohup bash thai/cs/run10.sh > thai/out/run10.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
mkdir -p thai/out

echo "== [$(date +%H:%M)] 1/4 Pantip-only items from cs9; cascade stopped"
docker stop laya-cascade >/dev/null 2>&1 || true
"${RUN[@]}" python filter_items.py --inp /work/thai/data/cs/cs9_items.pt --out /work/thai/data/cs/pantip9_items.pt --prefix pantip

echo "== [$(date +%H:%M)] 2/4 train from run 8 on the Pantip items, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run8 --items /work/thai/data/cs/pantip9_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run10

echo "== [$(date +%H:%M)] 3/4 eval: real Pantip set, cs9 held-out, public set"
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run10 --eval /work/thai/data/domain/real_cs_eval.jsonl --out /work/thai/out/run10_realcs8.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run10 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run10_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run10 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run10.json

echo "== [$(date +%H:%M)] 4/4 restore :8011 (run 8) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
