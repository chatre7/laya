#!/usr/bin/env bash
# Run 12, unattended: Google Play reviews labelled by teacher+student agreement (+ run 11 replay) -> a pass from run 11 ->
# evals (real long/short, hand-labelled reviews if the labels exist, next best action, cs9, public) -> restore :8011.
#   nohup bash thai/cs/run12.sh > thai/out/run12.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
mkdir -p thai/out
docker rm -f laya-run8 laya-run10 laya-student vllm-plan >/dev/null 2>&1 || true

echo "== [$(date +%H:%M)] 1/4 labels: student run 11 + teacher, agreement; items with replay (cascade stopped from here)"
docker stop laya-cascade >/dev/null 2>&1 || true
"${RUN[@]}" python label_play.py --teacher http://172.18.72.145:8010 --student /work/thai/out/laya-th-run11 --workers 12 --limit 12000

echo "== [$(date +%H:%M)] 2/4 train run 12 from run 11, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run11 --items /work/thai/data/cs/cs12_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run12

echo "== [$(date +%H:%M)] 3/4 evals"
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run12 --out /work/thai/out/real12.json
if [ -s thai/data_domain/play_labels.json ]; then
  "${RUN[@]}" python eval_play.py --models /work/thai/out/laya-th-run11,/work/thai/out/laya-th-run12 --out /work/thai/out/play12.json
fi
for biz in banking telecom insurance; do
  "${RUN[@]}" python eval_nba.py --business $biz --models "" --rule-model /work/thai/out/laya-th-run12 --teacher "" \
    --out /work/thai/out/nba_${biz}_run12.json 2>&1 | grep -v -iE 'warn|Fetching'
done
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run12 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run12_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run12 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run12.json

echo "== [$(date +%H:%M)] 4/4 restore :8011 (run 8) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
