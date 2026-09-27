#!/usr/bin/env bash
# Run 8, unattended: real Thai in-domain text. Run 7 records (intent lists extended) + banking77 / insurance-qa Thai rewrites
# (teacher-gated) + 932 hand-labelled Pantip questions (repeated). Trains from run 3, evals run 7 / run 8 on the cs8 held-out set,
# on the real Pantip set (real_cs_eval.jsonl, rebuilt with the extended lists) and run 8 on the public set; restores :8011 (run 7).
#   nohup bash thai/cs/run8.sh > thai/out/run8.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=172.18.72.145
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
CPU=(docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train)
mkdir -p thai/out

echo "== [$(date +%H:%M)] 1/5 label: cs7 records (e-commerce capped) + banking77/insurance-qa (teacher-gated) + Pantip hand labels; cascade stopped"
docker stop laya-cascade >/dev/null 2>&1 || true
"${CPU[@]}" python label_cs8.py --from-records cs7 --teacher http://$HOST:8010 --workers 8 --prefix cs8 \
  --min-p 0.3 --ecom-cap 30000 --pantip-repeat 6 --questions-per-record 3 --smooth 0.1

echo "== [$(date +%H:%M)] 2/5 train from run 3 (cs8_items + human items, 2 epochs)"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run3 --items /work/thai/data/cs/cs8_items.pt,/work/thai/data/train_items768.pt \
  --head-max-len 768 --epochs 2 --out /work/thai/out/laya-th-run8

echo "== [$(date +%H:%M)] 3/5 eval: cs8 held-out (run 7 / run 8)"
for r in run7 run8; do
  "${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-$r --eval /work/thai/data/cs/cs8_eval_human.jsonl \
    --teacher /work/thai/data/cs/cs8_eval.jsonl --out /work/thai/out/${r}_cs8.json
done

echo "== [$(date +%H:%M)] 4/5 eval: real Pantip set with the run 8 intent lists (run 3 / run 7 / run 8), public set (run 8)"
for r in run3 run7 run8; do
  "${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-$r --eval /work/thai/data/domain/real_cs_eval.jsonl \
    --out /work/thai/out/${r}_realcs8.json
done
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run8 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run8.json

echo "== [$(date +%H:%M)] 5/5 restore :8011 (still run 7) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
