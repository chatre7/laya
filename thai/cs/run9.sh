#!/usr/bin/env bash
# Run 9, unattended: run 8's recipe with a second hand-labelled Pantip batch (telecom 500 + banking 500 = 1,932 real rows in
# total). Same data otherwise, trained from run 3 like run 8 so the effect of the extra 1,000 rows is isolated. Evals: cs9 held-out
# (run 8 / run 9), the real Pantip set (run 8 / run 9), public set (run 9), krathu-500 (run 9). Restores :8011 (run 8).
#   nohup bash thai/cs/run9.sh > thai/out/run9.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=172.18.72.145
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
CPU=(docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train)
mkdir -p thai/out

echo "== [$(date +%H:%M)] 1/5 label: cs7 records + banking77/insurance-qa (gated) + Pantip extra + extra2; cascade stopped"
docker stop laya-cascade >/dev/null 2>&1 || true
"${CPU[@]}" python label_cs8.py --from-records cs7 --teacher http://$HOST:8010 --workers 8 --prefix cs9 \
  --pantip-sets extra,extra2 --min-p 0.3 --ecom-cap 30000 --pantip-repeat 6 --questions-per-record 3 --smooth 0.1

echo "== [$(date +%H:%M)] 2/5 train from run 3 (cs9_items + human items, 2 epochs)"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run3 --items /work/thai/data/cs/cs9_items.pt,/work/thai/data/train_items768.pt \
  --head-max-len 768 --epochs 2 --out /work/thai/out/laya-th-run9

echo "== [$(date +%H:%M)] 3/5 eval: cs9 held-out (run 8 / run 9)"
for r in run8 run9; do
  "${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-$r --eval /work/thai/data/cs/cs9_eval_human.jsonl \
    --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/${r}_cs9.json
done

echo "== [$(date +%H:%M)] 4/5 eval: real Pantip set (run 9), public set (run 9), krathu-500 (run 9)"
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run9 --eval /work/thai/data/domain/real_cs_eval.jsonl --out /work/thai/out/run9_realcs8.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run9 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run9 --eval /work/thai/data/domain/krathu500_eval.jsonl --out /work/thai/out/run9_krathu.json

echo "== [$(date +%H:%M)] 5/5 restore :8011 (run 8) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
