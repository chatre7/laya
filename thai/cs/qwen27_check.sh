#!/usr/bin/env bash
# Qwen3.8-27B at 4 bits (RedHatAI/Qwen3.8-27B-INT4, 19.5 GB) as a labelling teacher for the new questions, scored on the
# 307 hand labels like Saluki (the same model at 2 bits on llama.cpp). It needs both A2s: the teacher (:8010) and the
# cascade (:8011) are stopped for the run and restarted on exit, whatever happens.
#   nohup bash thai/cs/qwen27_check.sh > thai/out/qwen27_check.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=RedHatAI/Qwen3.8-27B-INT4
NAME=qwen3-27b
mkdir -p thai/out
echo "== [$(date +%H:%M)] 1/4 download (once)"
ls -d ~/.cache/huggingface/hub/models--${MODEL//\//--}/snapshots/*/config.json >/dev/null 2>&1 || \
  docker run --rm -e HF_HOME=/root/.cache/huggingface -v ~/.cache/huggingface:/root/.cache/huggingface laya-train \
    hf download "$MODEL" 2>&1 | grep -v -iE "warn|it/s" | tail -1
ls -d ~/.cache/huggingface/hub/models--${MODEL//\//--}/snapshots/*/config.json >/dev/null 2>&1 || { echo "download failed"; exit 1; }
du -sh ~/.cache/huggingface/hub/models--${MODEL//\//--} | cut -f1

restore() {
  docker rm -f vllm-27b >/dev/null 2>&1 || true
  docker start openthai-systemone laya-cascade >/dev/null 2>&1 || true
  for i in $(seq 1 60); do curl -s -m 5 -o /dev/null -w "%{http_code}" http://localhost:8011/healthz | grep -q 200 && break; sleep 5; done
  echo "== [$(date +%H:%M)] services back: $(docker ps --format '{{.Names}} {{.Status}}' | grep -E 'cascade|openthai' | tr '\n' ';')"
}
trap restore EXIT
echo "== [$(date +%H:%M)] 2/4 stop :8010 and :8011, start vLLM on both cards"
docker stop laya-cascade openthai-systemone >/dev/null 2>&1 || true
docker rm -f vllm-27b >/dev/null 2>&1 || true
docker run -d --name vllm-27b --gpus all --ipc=host -p 8012:8000 -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_SAMPLER=0 vllm/vllm-openai:latest --model "$MODEL" \
  --served-model-name "$NAME" --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --max-model-len 4096 --max-num-seqs 16 >/dev/null
up=0
for i in $(seq 1 240); do
  curl -s -m 3 http://localhost:8012/v1/models | grep -q "$NAME" && { echo "== [$(date +%H:%M)] $NAME up after $((i * 5)) s"; up=1; break; }
  docker ps --format '{{.Names}}' | grep -q vllm-27b || break
  sleep 5
done
[ "$up" = 1 ] || { echo "$NAME did not come up"; docker logs --tail 40 vllm-27b 2>&1 | cut -c1-220; exit 1; }
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

echo "== [$(date +%H:%M)] 3/4 label the 307 hand-checked texts"
time python3 thai/cs/label_new_gen.py --eval --url http://localhost:8012 --model "$NAME" --out thai/data/cs/new_gen27.jsonl --workers 16
docker rm -f vllm-27b >/dev/null 2>&1 || true

echo "== [$(date +%H:%M)] 4/4 score against the hand labels (Saluki and Qwen3-8B for reference)"
docker run --rm -v "$PWD":/work -w /work/thai/cs laya-train python eval_new.py --teacher /work/thai/data/cs/new_gen27.eval.jsonl 2>&1 | grep -v -iE 'warn|Fetching'
echo "--- Saluki (2-bit) for reference:"
docker run --rm -v "$PWD":/work -w /work/thai/cs laya-train python eval_new.py --teacher /work/thai/data/cs/new_saluki.eval.jsonl 2>&1 | grep -v -iE 'warn|Fetching' | grep -A4 "^== teacher"
echo "--- Qwen3-8B for reference:"
docker run --rm -v "$PWD":/work -w /work/thai/cs laya-train python eval_new.py --teacher /work/thai/data/cs/new_gen8.eval.jsonl 2>&1 | grep -v -iE 'warn|Fetching' | grep -A4 "^== teacher"
echo "== [$(date +%H:%M)] done"
