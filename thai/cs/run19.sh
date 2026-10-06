#!/usr/bin/env bash
# Run 19, unattended: three new questions (churn threat, threat to go outside, contact effort), taught by Qwen3-8B reading
# written rules (label_new_gen.py), checked first against 307 hand labels (new_check.sh, eval_new.py).
# Labels over the whole pool (the cascade stays up) -> items -> a pass from run 16 (the cascade is down from here) ->
# evals: the new questions on the hand-checked texts, then everything run 16 was measured on -> restore :8011 (run 16).
#   nohup bash thai/cs/run19.sh > thai/out/run19.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g -e PYTHONUNBUFFERED=1 laya-train)
mkdir -p thai/out

echo "== [$(date +%H:%M)] 1/5 labels (Qwen3-8B beside the cascade)"
bash thai/cs/vllm_up.sh 8b 0.76 16 || { echo "labeller did not start"; exit 1; }
python3 thai/cs/label_new_gen.py --model qwen3-8b --out thai/data/cs/new_gen8.jsonl --workers 12 || { echo "labelling failed"; exit 1; }
docker rm -f vllm-plan >/dev/null 2>&1 || true
[ "$(wc -l < thai/data/cs/new_gen8.jsonl)" -gt 20000 ] || { echo "too few labels, stopping before the cascade is touched"; exit 1; }

set -e
docker stop laya-cascade >/dev/null 2>&1 || true
trap 'docker start laya-cascade >/dev/null 2>&1 || true; echo "== [$(date +%H:%M)] cascade restarted (run 16)"' EXIT
echo "== [$(date +%H:%M)] 2/5 items"
"${RUN[@]}" python label_cs19.py

echo "== [$(date +%H:%M)] 3/5 train run 19 from run 16, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run16 --items /work/thai/data/cs/cs19_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run19

echo "== [$(date +%H:%M)] 4/5 evals"
set +e
"${RUN[@]}" python eval_new.py --teacher /work/thai/data/cs/new_gen8.eval.jsonl --models /work/thai/out/laya-th-run16,/work/thai/out/laya-th-run19 \
  --out /work/thai/out/new19.json 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python new_questions_demo.py --models /work/thai/out/laya-th-run19 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python eval_stars.py --models /work/thai/out/laya-th-run16,/work/thai/out/laya-th-run19 --n 3000 --out /work/thai/out/stars19.json 2>&1 | grep -v -iE 'warn|Fetching'
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run19 --out /work/thai/out/real19.json
"${RUN[@]}" python eval_play.py --models /work/thai/out/laya-th-run19 --out /work/thai/out/play19.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run19 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run19_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run19 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run19.json

echo "== [$(date +%H:%M)] 5/5 permissions; the cascade restarts on exit"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
echo "== [$(date +%H:%M)] done"
