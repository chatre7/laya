#!/usr/bin/env bash
# Run 6, unattended: four-business customer-service student (telecom, banking, insurance, e-commerce) + out-of-scope.
# Waits for prep_cs_data.sh to finish (prep.log "done"), then: second teacher on GPU 1, label with two teachers, train from
# run 3, eval on the cs held-out set (run 3 / run 5 / run 6) and the public set, restore :8011 (still run 5).
#   nohup bash thai/cs/run6.sh > thai/out/run6.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=172.18.72.145
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
CPU=(docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train)
mkdir -p thai/out

echo "== [$(date +%H:%M)] 0/5 waiting for prep_cs_data.sh"
while ! grep -q "== \[.*\] done" thai/data/cs/prep.log 2>/dev/null; do sleep 60; done
[ -s thai/data/cs/cs_colloquial_checked.jsonl ] || { echo "no checked rewrites"; exit 1; }
echo "== [$(date +%H:%M)] prep done: $(wc -l < thai/data/cs/cs_colloquial_checked.jsonl) checked rewrites"

echo "== [$(date +%H:%M)] 1/5 second teacher on GPU 1 (:8013), cascade stopped"
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
docker stop laya-cascade >/dev/null 2>&1 || true
docker rm -f ots-gpu1 >/dev/null 2>&1 || true
docker run -d --name ots-gpu1 --gpus '"device=1"' -p 8013:8000 -v docker_hf-cache:/hf \
  -e OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne -e OTS_MAX_BATCH=16 -e OTS_MAX_BATCH_TOKENS=8192 -e OTS_MAX_WAIT_MS=8 \
  -e OTS_MAX_QUEUE=256 -e OTS_MAX_QUEUE_WAIT_S=10 -e OTS_MAX_STATE_TOKENS=4096 openthai-systemone:0.1.0-batched >/dev/null
for i in $(seq 1 60); do curl -sf -m 3 http://localhost:8013/healthz >/dev/null && break; sleep 5; done
curl -sf -m 3 http://localhost:8013/healthz || { echo "second teacher did not come up"; docker logs ots-gpu1 | tail -20; exit 1; }
echo

echo "== [$(date +%H:%M)] 2/5 label (cs + e-commerce + wisesight) with two teachers"
"${CPU[@]}" python label_cs.py --teacher http://$HOST:8010,http://$HOST:8013 --workers 16 --questions-per-record 3 --smooth 0.1 --prefix cs6
docker rm -f ots-gpu1 >/dev/null

echo "== [$(date +%H:%M)] 3/5 train from run 3 (cs6_items + human items, 2 epochs)"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run3 --items /work/thai/data/cs/cs6_items.pt,/work/thai/data/train_items768.pt \
  --head-max-len 768 --epochs 2 --out /work/thai/out/laya-th-run6

echo "== [$(date +%H:%M)] 4/5 eval: cs held-out (run 3 / run 5 / run 6), public set (run 6)"
for r in run3 run5 run6; do
  "${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-$r --eval /work/thai/data/cs/cs6_eval_human.jsonl \
    --teacher /work/thai/data/cs/cs6_eval.jsonl --out /work/thai/out/${r}_cs6.json
done
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run6 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run6.json

echo "== [$(date +%H:%M)] 5/5 restore :8011 (still run 5) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
