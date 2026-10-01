#!/usr/bin/env bash
# Run 14, unattended: Qwen3-8B labels the Pantip / wisesight posts that have no hand label; items = posts where the LLM and
# run 11 agree + capped LLM-labelled reviews + run 11 replay -> a pass from run 11 -> evals -> restore :8011.
#   nohup bash thai/cs/run14.sh > thai/out/run14.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
mkdir -p thai/out
docker stop laya-cascade >/dev/null 2>&1 || true

echo "== [$(date +%H:%M)] 1/5 LLM labels for the unlabelled posts (Qwen3-8B-FP8)"
if ! curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-8b; then
  docker rm -f vllm-plan >/dev/null 2>&1 || true
  docker run -d --name vllm-plan --gpus '"device=1"' --ipc=host -p 8012:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 vllm/vllm-openai:latest --model RedHatAI/Qwen3-8B-FP8-dynamic \
    --served-model-name qwen3-8b --gpu-memory-utilization 0.92 --max-model-len 4096 --max-num-seqs 32 >/dev/null
  for i in $(seq 1 60); do curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-8b && break; sleep 5; done
fi
python3 thai/cs/label_pantip_llm.py --out thai/data/cs/pantip_llm.jsonl
docker rm -f vllm-plan >/dev/null 2>&1 || true

echo "== [$(date +%H:%M)] 2/5 items (agreement with run 11 on the posts, capped reviews, replay)"
"${RUN[@]}" python label_cs14.py

echo "== [$(date +%H:%M)] 3/5 train run 14 from run 11, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run11 --items /work/thai/data/cs/cs14_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run14

echo "== [$(date +%H:%M)] 4/5 evals"
"${RUN[@]}" python eval_play.py --models /work/thai/out/laya-th-run14 --out /work/thai/out/play14.json
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run14 --out /work/thai/out/real14.json
for biz in banking telecom insurance; do
  "${RUN[@]}" python eval_nba.py --business $biz --models "" --rule-model /work/thai/out/laya-th-run14 --teacher "" \
    --out /work/thai/out/nba_${biz}_run14.json 2>&1 | grep -v -iE 'warn|Fetching'
done
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run14 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run14_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run14 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run14.json

echo "== [$(date +%H:%M)] 5/5 restore :8011 (run 8) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
