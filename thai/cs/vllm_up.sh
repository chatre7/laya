#!/usr/bin/env bash
# Start the generative labeller on GPU 1 at :8012 (vLLM, FP8 Qwen3 from the local HF cache) and wait until it answers.
#   bash thai/cs/vllm_up.sh 4b     # 0.45 of the card, fits beside the cascade
#   bash thai/cs/vllm_up.sh 8b     # 0.92 of the card: stop laya-cascade first
#   bash thai/cs/vllm_up.sh 8b 0.76 16   # the 8B squeezed in beside the cascade: less KV cache, 16 parallel requests
#   bash thai/cs/vllm_up.sh 3.5-9b 0.76 16
set -uo pipefail
SIZE=${1:-4b}
case "$SIZE" in
  8b) MODEL=RedHatAI/Qwen3-8B-FP8-dynamic; UTIL=0.92 ;;
  3.5-9b) MODEL=RedHatAI/Qwen3.5-9B-FP8-dynamic; UTIL=0.95 ;;        # 14 GB with its vision tower and 248k vocab: the card alone, and barely
  3.5-9b-w4) MODEL=RedHatAI/Qwen3.5-9B-quantized.w4a16; UTIL=0.76 ;;  # the 4-bit one fits beside the cascade
  *) MODEL=RedHatAI/Qwen3-4B-FP8-dynamic; UTIL=0.45 ;;
esac
# the vLLM container runs offline: fetch the weights into the shared cache first when they are not there yet
ls -d ~/.cache/huggingface/hub/models--${MODEL//\//--} >/dev/null 2>&1 ||   docker run --rm -e HF_HOME=/root/.cache/huggingface -v ~/.cache/huggingface:/root/.cache/huggingface laya-train hf download "$MODEL" 2>&1 | grep -v -iE "warn|it/s" | tail -1
UTIL=${2:-$UTIL}
SEQS=${3:-32}
if curl -s -m 3 http://localhost:8012/v1/models | grep -q "qwen3-$SIZE"; then echo "qwen3-$SIZE already up"; exit 0; fi
docker rm -f vllm-plan >/dev/null 2>&1 || true
docker run -d --name vllm-plan --gpus '"device=1"' --ipc=host -p 8012:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 vllm/vllm-openai:latest --model "$MODEL" \
  --served-model-name "qwen3-$SIZE" --gpu-memory-utilization "$UTIL" --max-model-len 4096 --max-num-seqs "$SEQS" >/dev/null
for i in $(seq 1 90); do curl -s -m 3 http://localhost:8012/v1/models | grep -q "qwen3-$SIZE" && { echo "qwen3-$SIZE up after $((i * 5)) s"; exit 0; }; sleep 5; done
echo "qwen3-$SIZE did not come up"; docker logs --tail 20 vllm-plan 2>&1 | cut -c1-200; exit 1
