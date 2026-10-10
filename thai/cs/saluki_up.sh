#!/usr/bin/env bash
# Underdog-Saluki-27B (Qwen3.8-27B, 2-bit GGUF, 7.9 GB) on llama.cpp at :8013, GPU 1, beside the cascade. A candidate
# labelling teacher for the new questions, to be scored by label_new_gen.py --eval against the hand labels.
#   bash thai/cs/saluki_up.sh            # download (once), pull the llama.cpp image (once), start, wait
set -uo pipefail
MODEL_DIR=~/models/saluki
GGUF=Underdog-Saluki-27B-1.0-IQ2-mix.gguf
IMG=ghcr.io/ggml-org/llama.cpp:server-cuda
mkdir -p "$MODEL_DIR"
if [ ! -s "$MODEL_DIR/$GGUF" ]; then
  echo "== [$(date +%H:%M)] download $GGUF"
  docker run --rm -v "$MODEL_DIR":/dl -v ~/.cache/huggingface:/root/.cache/huggingface laya-train \
    hf download ConwayResearch/Underdog-Saluki-27B-1.0 "$GGUF" --local-dir /dl 2>&1 | grep -v -iE "warn|it/s" | tail -2
fi
ls -la "$MODEL_DIR/$GGUF" || { echo "download failed"; exit 1; }
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "== [$(date +%H:%M)] pull $IMG"; docker pull -q "$IMG" | tail -1; }
docker rm -f llama-saluki >/dev/null 2>&1 || true
docker run -d --name llama-saluki --gpus '"device=1"' -p 8013:8080 -v "$MODEL_DIR":/models "$IMG" \
  -m "/models/$GGUF" --jinja -ngl 99 -fa on -c 16384 -np 4 --host 0.0.0.0 --port 8080 >/dev/null
for i in $(seq 1 60); do curl -s -m 3 http://localhost:8013/v1/models | grep -q "model" && { echo "== [$(date +%H:%M)] saluki up after $((i * 5)) s"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; exit 0; }; sleep 5; done
echo "saluki did not come up"; docker logs --tail 30 llama-saluki 2>&1 | cut -c1-200; exit 1
