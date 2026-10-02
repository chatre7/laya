#!/usr/bin/env bash
# Run 15, unattended: mood on short service text incl. sarcasm. Needs thai/data/cs/sarcasm_gen.jsonl (gen_sarcasm.py) and
# play_sentiment.jsonl (mine_sarcasm.py). Items -> a pass from run 14 -> evals -> restore :8011 (still run 14).
#   nohup bash thai/cs/run15.sh > thai/out/run15.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
mkdir -p thai/out
docker stop laya-cascade >/dev/null 2>&1 || true
docker rm -f vllm-plan >/dev/null 2>&1 || true

echo "== [$(date +%H:%M)] 1/4 items"
"${RUN[@]}" python label_cs15.py

echo "== [$(date +%H:%M)] 2/4 train run 15 from run 14, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run14 --items /work/thai/data/cs/cs15_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run15

echo "== [$(date +%H:%M)] 3/4 evals"
"${RUN[@]}" python eval_stars.py --models /work/thai/out/laya-th-run14,/work/thai/out/laya-th-run15 --n 3000 --out /work/thai/out/stars15.json 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run15 --out /work/thai/out/real15.json
"${RUN[@]}" python eval_play.py --models /work/thai/out/laya-th-run15 --out /work/thai/out/play15.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run15 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run15_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run15 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run15.json

echo "== [$(date +%H:%M)] 4/4 restore :8011 (run 14) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
