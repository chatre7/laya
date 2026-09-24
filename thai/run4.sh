#!/usr/bin/env bash
# Run 4, unattended: call-center distillation. Second teacher on GPU 1 for labelling (the cascade and the rewrite vLLM
# are stopped for the duration), then train from the run 3 checkpoint on cc_items + the human-labelled items, eval on
# the held-out call-center set (run 3 evaluated on the same set as baseline) and on the old eval set, then bring :8011 back.
#   nohup bash thai/run4.sh > thai/out/run4.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=172.18.72.145
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf --shm-size 2g laya-train)
CPU=(docker run --rm -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf laya-train)
mkdir -p thai/out

echo "== [$(date +%H:%M)] 0/5 free GPU 1 (vllm-rewrite, laya-cascade) and start a second teacher on it (:8013)"
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
docker stop laya-cascade >/dev/null 2>&1 || true
docker rm -f ots-gpu1 >/dev/null 2>&1 || true
docker run -d --name ots-gpu1 --gpus '"device=1"' -p 8013:8000 -v docker_hf-cache:/hf \
  -e OPENTHAI_SYSTEMONE_MODEL=iapp/OpenThai-SystemOne -e OTS_MAX_BATCH=16 -e OTS_MAX_BATCH_TOKENS=8192 -e OTS_MAX_WAIT_MS=8 \
  -e OTS_MAX_QUEUE=256 -e OTS_MAX_QUEUE_WAIT_S=10 -e OTS_MAX_STATE_TOKENS=4096 openthai-systemone:0.1.0-batched >/dev/null
for i in $(seq 1 60); do curl -sf -m 3 http://localhost:8013/healthz >/dev/null && break; sleep 5; done
curl -sf -m 3 http://localhost:8013/healthz || { echo "second teacher did not come up"; docker logs ots-gpu1 | tail -20; exit 1; }
echo

echo "== [$(date +%H:%M)] 1/5 label with two teachers"
"${CPU[@]}" python label_cc.py --teacher http://$HOST:8010,http://$HOST:8013 --workers 16 --wisesight 12000 --questions-per-record 2

echo "== [$(date +%H:%M)] 2/5 stop the second teacher, train from run 3 (cc_items + human items, 2 epochs)"
docker rm -f ots-gpu1 >/dev/null
"${RUN[@]}" python train_single.py --model /work/thai/out/laya-th-run3 --items /work/thai/data/cc/cc_items.pt,/work/thai/data/train_items768.pt \
  --head-max-len 768 --epochs 2 --out /work/thai/out/laya-th-run4

echo "== [$(date +%H:%M)] 3/5 eval on the held-out call-center set: run 3 (baseline) then run 4"
"${RUN[@]}" python eval_thai.py --model /work/thai/out/laya-th-run3 --eval /work/thai/data/cc/cc_eval_human.jsonl \
  --teacher /work/thai/data/cc/cc_eval.jsonl --out /work/thai/out/run3_cc.json
"${RUN[@]}" python eval_thai.py --model /work/thai/out/laya-th-run4 --eval /work/thai/data/cc/cc_eval_human.jsonl \
  --teacher /work/thai/data/cc/cc_eval.jsonl --out /work/thai/out/run4_cc.json

echo "== [$(date +%H:%M)] 4/5 eval on the old public set (regression check)"
"${RUN[@]}" python eval_thai.py --model /work/thai/out/laya-th-run4 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run4.json

echo "== [$(date +%H:%M)] 5/5 restore :8011 (still run 3) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
