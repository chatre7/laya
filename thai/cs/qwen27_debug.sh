#!/usr/bin/env bash
# Second attempt at RedHatAI/Qwen3.8-27B-INT4 on vLLM with the full engine log kept (qwen27_check.sh only kept the tail,
# which hid the root cause). Services stopped and restored as in qwen27_check.sh.
#   nohup bash thai/cs/qwen27_debug.sh > thai/out/qwen27_debug.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=RedHatAI/Qwen3.8-27B-INT4
restore() {
  docker logs vllm-27b > thai/out/vllm27_full.log 2>&1 || true
  docker rm -f vllm-27b >/dev/null 2>&1 || true
  docker start openthai-systemone laya-cascade >/dev/null 2>&1 || true
  for i in $(seq 1 60); do curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:8011/healthz | grep -q 200 && break; sleep 5; done
  echo "== [$(date +%H:%M)] services back: $(docker ps --format '{{.Names}} {{.Status}}' | grep -E 'cascade|openthai' | tr '\n' ';')"
}
trap restore EXIT
docker stop laya-cascade openthai-systemone >/dev/null 2>&1 || true
docker rm -f vllm-27b >/dev/null 2>&1 || true
echo "== [$(date +%H:%M)] start ($*)"
docker run -d --name vllm-27b --gpus all --ipc=host -p 8012:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 vllm/vllm-openai:latest --model "$MODEL" \
  --served-model-name qwen3-27b --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 4096 --max-num-seqs 16 "$@" >/dev/null
for i in $(seq 1 180); do
  curl -s -m 3 http://localhost:8012/v1/models | grep -q qwen3-27b && { echo "== [$(date +%H:%M)] up after $((i * 5)) s"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 0; }
  docker ps --format '{{.Names}}' | grep -q vllm-27b || { echo "== [$(date +%H:%M)] container exited after $((i * 5)) s"; break; }
  sleep 5
done
echo "== root cause candidates:"
docker logs vllm-27b 2>&1 | grep -nE "Error|error|OOM|out of memory|not support|Unsupported|ValueError|AssertionError|KeyError|NotImplemented|CUDA|free memory|Available KV" | grep -vE "Traceback|raise RuntimeError|Engine core init" | cut -c1-300 | head -25
exit 1
