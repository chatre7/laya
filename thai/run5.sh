#!/usr/bin/env bash
# Run 5, unattended, from the run 4 labelled records (no teacher labelling): 3 questions per text, label smoothing 0.1 on the
# human one-hot targets, an `other` option on the intent question (wisesight texts = out-of-scope examples), trained from the
# run 3 checkpoint. Then eval run 3 / run 4 / run 5 on the new held-out set, run 5 on the public set, cascade sweeps, and :8011 back.
#   nohup bash thai/run5.sh > thai/out/run5.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf --shm-size 2g laya-train)
CPU=(docker run --rm -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf laya-train)
mkdir -p thai/out

echo "== [$(date +%H:%M)] 1/5 items from the labelled records (3 q/record, smooth 0.1, intent+other)"
docker stop laya-cascade >/dev/null 2>&1 || true
"${CPU[@]}" python label_cc.py --from-records --prefix cc5 --questions-per-record 3 --smooth 0.1 --intent-other

echo "== [$(date +%H:%M)] 2/5 train from run 3 (cc5_items + human items, 2 epochs)"
"${RUN[@]}" python train_single.py --model /work/thai/out/laya-th-run3 --items /work/thai/data/cc/cc5_items.pt,/work/thai/data/train_items768.pt \
  --head-max-len 768 --epochs 2 --out /work/thai/out/laya-th-run5

echo "== [$(date +%H:%M)] 3/5 eval on the new held-out set (intent has other): run 3, run 4, run 5"
for r in run3 run4 run5; do
  "${RUN[@]}" python eval_thai.py --model /work/thai/out/laya-th-$r --eval /work/thai/data/cc/cc5_eval_human.jsonl \
    --teacher /work/thai/data/cc/cc5_eval.jsonl --out /work/thai/out/${r}_cc5.json
done

echo "== [$(date +%H:%M)] 4/5 eval run 5 on the public set + cascade sweeps"
"${RUN[@]}" python eval_thai.py --model /work/thai/out/laya-th-run5 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run5.json
"${RUN[@]}" python cascade.py --student /work/thai/out/laya-th-run5 --eval /work/thai/data/cc/cc5_eval_human.jsonl --out /work/thai/out/cascade5_cc.json --max-options 60
"${RUN[@]}" python cascade.py --student /work/thai/out/laya-th-run5 --eval /work/thai/data/eval.jsonl --out /work/thai/out/cascade5.json --max-options 60

echo "== [$(date +%H:%M)] 5/5 restore :8011 (still run 4) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
