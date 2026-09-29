#!/usr/bin/env bash
# Run 11, unattended: short chat versions of the real Pantip posts (Qwen3-4B), human frustration labels, hard examples
# (in-scope rows run 8 calls `other`), replay -> a second pass from run 8 like run 10 -> evals (real long + short + frustration,
# next best action for the three businesses, cs9 held-out, public set) -> restore :8011.
#   nohup bash thai/cs/run11.sh > thai/out/run11.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
RUN=(docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g laya-train)
mkdir -p thai/out
docker rm -f laya-run8 laya-run10 laya-student >/dev/null 2>&1 || true

if [ ! -s thai/data/cs/pantip_short.jsonl ]; then
  echo "== [$(date +%H:%M)] 1/6 short chat versions (Qwen3-4B-FP8 on GPU 1, :8012)"
  docker rm -f vllm-plan >/dev/null 2>&1 || true
  docker run -d --name vllm-plan --gpus '"device=1"' --ipc=host -p 8012:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 vllm/vllm-openai:latest --model RedHatAI/Qwen3-4B-FP8-dynamic \
    --served-model-name qwen3-4b --gpu-memory-utilization 0.45 --max-model-len 4096 --max-num-seqs 32 >/dev/null
  for i in $(seq 1 60); do curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-4b && break; sleep 5; done
  python3 thai/cs/shorten_pantip.py --domain thai/data_domain --out thai/data/cs/pantip_short.jsonl
  docker rm -f vllm-plan >/dev/null 2>&1 || true
fi

echo "== [$(date +%H:%M)] 2/6 baselines on the real sets (run 8, run 10)"
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run8,/work/thai/out/laya-th-run10 --out /work/thai/out/real_baselines.json

echo "== [$(date +%H:%M)] 3/6 items (cascade stopped from here)"
docker stop laya-cascade >/dev/null 2>&1 || true
"${RUN[@]}" python label_cs11.py --hard-model /work/thai/out/laya-th-run8

echo "== [$(date +%H:%M)] 4/6 train run 11 from run 8, 2 epochs, LR 1e-5 / 4e-5"
"${RUN[@]}" python ../train_single.py --model /work/thai/out/laya-th-run8 --items /work/thai/data/cs/cs11_items.pt \
  --head-max-len 768 --epochs 2 --lr-encoder 1e-5 --lr-head 4e-5 --out /work/thai/out/laya-th-run11

echo "== [$(date +%H:%M)] 5/6 evals"
"${RUN[@]}" python eval_real.py --models /work/thai/out/laya-th-run11 --out /work/thai/out/real11.json
for biz in banking telecom insurance; do
  "${RUN[@]}" python eval_nba.py --business $biz --models "" --rule-model /work/thai/out/laya-th-run11 --teacher "" \
    --out /work/thai/out/nba_${biz}_run11.json 2>&1 | grep -v -iE 'warn|Fetching'
done
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run11 --eval /work/thai/data/cs/cs9_eval_human.jsonl \
  --teacher /work/thai/data/cs/cs9_eval.jsonl --out /work/thai/out/run11_cs9.json
"${RUN[@]}" python ../eval_thai.py --model /work/thai/out/laya-th-run11 --eval /work/thai/data/eval.jsonl \
  --teacher /work/thai/data/distill3_eval.jsonl --out /work/thai/out/run11.json

echo "== [$(date +%H:%M)] 6/6 restore :8011 (run 8) and permissions"
"${RUN[@]}" chmod -R a+rX /work/thai/out /work/thai/data
docker start laya-cascade >/dev/null || true
echo "== [$(date +%H:%M)] done"
