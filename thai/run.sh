#!/usr/bin/env bash
# Laya Thai fine-tune experiment on the dev GPU box (GPU 1, the serving container keeps GPU 0).
# Layout on the host: ~/laya (this fork) mounted at /work; data/ and out/ live under thai/
#   bash run.sh build            build the laya-train image (serving image + datasets)
#   bash run.sh prep [LIMIT]     convert Thai datasets -> data/train_items.pt + data/eval.jsonl
#   bash run.sh eval MODEL OUT   evaluate a checkpoint (HF id or /work/thai/out/...) -> out/OUT.json
#   bash run.sh train [ARGS...]  fine-tune -> out/laya-th (runs in the background, log in out/train.log)
#   bash run.sh distill [ARGS]   label Thai texts with the OpenThai teacher -> data/distill_items.pt (background)
#   bash run.sh train-distill    fine-tune on the distillation set -> out/laya-th-distill (background)
#   bash run.sh eval-distill MODEL OUT   human eval set + student-vs-teacher agreement
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=laya-train
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf --shm-size 2g "$IMG")
case "${1:-}" in
  build) docker build -t "$IMG" thai ;;
  prep)  "${RUN[@]}" python prep_thai.py --out /work/thai/data --limit "${2:-1500}" ;;
  eval)  "${RUN[@]}" python eval_thai.py --model "$2" --eval /work/thai/data/eval.jsonl --out "/work/thai/out/$3.json" ;;
  train) shift; mkdir -p thai/out
         nohup "${RUN[@]}" python train_single.py --items /work/thai/data/train_items.pt --out /work/thai/out/laya-th "$@" > thai/out/train.log 2>&1 &
         echo "training started, pid $!, log: thai/out/train.log" ;;
  distill) shift; mkdir -p thai/out
         nohup "${RUN[@]}" python distill_from_ots.py --out /work/thai/data "$@" > thai/out/distill.log 2>&1 &
         echo "distillation labelling started, pid $!, log: thai/out/distill.log" ;;
  train-distill) shift; mkdir -p thai/out
         nohup "${RUN[@]}" python train_single.py --items /work/thai/data/distill_items.pt --out /work/thai/out/laya-th-distill "$@" > thai/out/train_distill.log 2>&1 &
         echo "distillation training started, pid $!, log: thai/out/train_distill.log" ;;
  eval-distill) "${RUN[@]}" python eval_thai.py --model "$2" --eval /work/thai/data/eval.jsonl --teacher /work/thai/data/distill_eval.jsonl --out "/work/thai/out/$3.json" ;;
  shell) "${RUN[@]}" bash ;;
  *) echo "usage: run.sh build|prep|eval|train|distill|train-distill|eval-distill|shell"; exit 1 ;;
esac
