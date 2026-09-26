#!/usr/bin/env bash
# Run 7, unattended: run 6 data + in-register out-of-scope texts (gen_other.sh) so `other` / `business` stop keying on register.
# Waits for gen_other.sh (gen_other.log "done"), reuses the cs6 labelled records (only the new texts go to the teacher), trains
# from run 3, evals run 5 / run 6 / run 7 on the re-cut held-out set and run 7 on the public set, restores :8011 (still run 6).
#   nohup bash thai/cs/run7.sh > thai/out/run7.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=172.18.72.145
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
CPU=(docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train)
mkdir -p thai/out

echo "== [$(date +%H:%M)] 0/5 waiting for gen_other.sh"
while ! grep -q "== \[.*\] done" thai/data/cs/gen_other.log 2>/dev/null; do sleep 60; done
[ -s thai/data/cs/other_indomain_kept.jsonl ] || { echo "no kept out-of-scope texts"; exit 1; }
echo "== [$(date +%H:%M)] gen done: $(wc -l < thai/data/cs/other_indomain_kept.jsonl) kept out-of-scope texts"

echo "== [$(date +%H:%M)] 1/5 label the new texts (cs6 records reused), cascade stopped"
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
docker stop laya-cascade >/dev/null 2>&1 || true
"${CPU[@]}" python label_cs.py --from-records cs6 --other-file /work/thai/data/cs/other_indomain_kept.jsonl \
  --teacher http://$HOST:8010 --workers 8 --questions-per-record 3 --smooth 0.1 --prefix cs7

echo "== [$(date +%H:%M)] 2/5 train from run 3 (cs7_items + human items, 2 epochs)"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run3 --items /work/thai/data/cs/cs7_items.pt,/work/thai/data/train_items768.pt \
  --head-max-len 768 --epochs 2 --out /work/thai/out/laya-th-run7

echo "== [$(date +%H:%M)] 3/5 eval: cs7 held-out (run 5 / run 6 / run 7)"
for r in run5 run6 run7; do
  "${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-$r --eval /work/thai/data/cs/cs7_eval_human.jsonl \
    --teacher /work/thai/data/cs/cs7_eval.jsonl --out /work/thai/out/${r}_cs7.json
done

echo "== [$(date +%H:%M)] 4/5 eval run 7 on the public set"
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run7 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run7.json

echo "== [$(date +%H:%M)] 5/5 restore :8011 (still run 6) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
