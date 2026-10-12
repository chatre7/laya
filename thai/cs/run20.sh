#!/usr/bin/env bash
# Run 20, unattended: the four new questions relabelled by Underdog-Saluki-27B over the cue-word texts (saluki_label.sh,
# already done), a pass from run 19 with the run 16 replay. The cascade (:8011, run 19) is down only while training; it is
# restarted before the evals, which run beside it on GPU 1.
#   nohup bash thai/cs/run20.sh > thai/out/run20.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g -e PYTHONUNBUFFERED=1 laya-train)
mkdir -p thai/out
[ "$(wc -l < thai/data/cs/new_saluki.jsonl)" -ge 4000 ] || { echo "Saluki labels missing, stopping before the cascade is touched"; exit 1; }

set -e
docker stop laya-cascade >/dev/null 2>&1 || true
trap 'docker start laya-cascade >/dev/null 2>&1 || true; echo "== [$(date +%H:%M)] cascade up (run 19)"' EXIT
echo "== [$(date +%H:%M)] 1/4 items"
"${RUN[@]}" python label_cs20.py

echo "== [$(date +%H:%M)] 2/4 train run 20 from run 19, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run19 --items /work/thai/data/cs/cs20_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run20

docker start laya-cascade >/dev/null 2>&1 || true
echo "== [$(date +%H:%M)] cascade up (run 19); 3/4 evals beside it"
set +e
"${RUN[@]}" python eval_new.py --teacher /work/thai/data/cs/new_saluki.eval.jsonl --models /work/thai/out/laya-th-run19,/work/thai/out/laya-th-run20 \
  --out /work/thai/out/new20.json 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python new_questions_demo.py --models /work/thai/out/laya-th-run20 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python eval_stars.py --models /work/thai/out/laya-th-run19,/work/thai/out/laya-th-run20 --n 3000 --out /work/thai/out/stars20.json 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run20 --out /work/thai/out/real20.json
"${RUN[@]}" python eval_play.py --models /work/thai/out/laya-th-run20 --out /work/thai/out/play20.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run20 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run20_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run20 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run20.json

echo "== [$(date +%H:%M)] 4/4 permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
echo "== [$(date +%H:%M)] done"
