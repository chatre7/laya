#!/usr/bin/env bash
# Customer-service (telecom / banking / insurance) training data, unattended:
#   1. vLLM (Qwen3-4B) on GPU 1 next to the cascade container
#   2. fetch + sample the three Bitext sets (300 per intent)
#   3. English utterance -> Thai colloquial customer message, 2 variants each
#   4. teacher gate: keep rewrites whose intent the teacher still recognises (per business)
#   nohup bash thai/cs/prep_cs_data.sh > thai/data/cs/prep.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
CPU=(docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train)
mkdir -p thai/data/cs

echo "== [$(date +%H:%M)] 1/4 vLLM qwen3-4b on GPU 1 (:8012)"
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
docker run -d --name vllm-rewrite --gpus '"device=1"' --ipc=host -p 8012:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 vllm/vllm-openai:latest \
  --model Qwen/Qwen3-4B --served-model-name qwen3-4b --dtype bfloat16 --gpu-memory-utilization 0.62 --max-model-len 4096 --max-num-seqs 64 >/dev/null

echo "== [$(date +%H:%M)] 2/4 fetch + sample Bitext telco / banking / insurance"
"${CPU[@]}" python fetch_bitext_cs.py --out /work/thai/data/cs --per-intent "${PER_INTENT:-300}"

for i in $(seq 1 120); do curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-4b && break; sleep 5; done
curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-4b || { echo "vLLM did not come up"; docker logs vllm-rewrite | tail -20; exit 1; }

echo "== [$(date +%H:%M)] 3/4 English -> Thai colloquial, ${VARIANTS:-2} variants per utterance"
python3 thai/cs/translate_colloquial.py --inp thai/data/cs/bitext_cs_en.jsonl --out thai/data/cs/cs_colloquial.jsonl --variants "${VARIANTS:-2}" --workers 32 --seed 42

echo "== [$(date +%H:%M)] 4/4 teacher gate (per-business intent question, :8010)"
docker rm -f vllm-rewrite >/dev/null 2>&1 || true
python3 thai/cs/check_cs_rewrites.py --inp thai/data/cs/cs_colloquial.jsonl --out thai/data/cs/cs_colloquial_checked.jsonl --teacher http://localhost:8010 --workers 8
echo "== [$(date +%H:%M)] done"
