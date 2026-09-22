#!/usr/bin/env bash
# Run 3, unattended: bigger distillation set (8,000 texts per source, wide-intent questions, order-invariant teacher answers disabled (5x slower labelling for little gain)
# teacher answers for >= 11 options), option budget 768, human-labelled items re-tokenised at 768 and mixed in,
# 2 epochs, then eval. Log: thai/out/run3.log
#   nohup bash thai/run3.sh > thai/out/run3.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=laya-train
HML=768
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf --shm-size 2g "$IMG")
mkdir -p thai/out

echo "== [$(date +%H:%M)] 1/4 distillation set (teacher on GPU 0)"
"${RUN[@]}" python distill_from_ots.py --out /work/thai/data --per-source 8000 --head-max-len $HML \
  --order-invariant-min-options 0 --prefix distill3 --workers 8

echo "== [$(date +%H:%M)] 2/4 human-labelled items at head_max_len=$HML"
"${RUN[@]}" python prep_thai.py --out /work/thai/data --limit 1500 --head-max-len $HML --items-name train_items768.pt

echo "== [$(date +%H:%M)] 3/4 train (distill3 + human, 2 epochs)"
"${RUN[@]}" python train_single.py --items /work/thai/data/distill3_items.pt,/work/thai/data/train_items768.pt \
  --head-max-len $HML --epochs 2 --out /work/thai/out/laya-th-run3

echo "== [$(date +%H:%M)] 4/4 eval"
"${RUN[@]}" python eval_thai.py --model /work/thai/out/laya-th-run3 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run3.json
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
echo "== [$(date +%H:%M)] done"
