#!/usr/bin/env bash
# Run one script of thai/d2 in the training image on GPU 1, beside the cascade (nothing is stopped).
#   bash thai/d2/run.sh smoke.py vllm-sr/Decision-2.0-Kai-0.6B
set -euo pipefail
cd "$(dirname "$0")/../.."
docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/d2 -v docker_hf-cache:/hf --shm-size 2g laya-train \
  python "$@" 2>&1 | grep -v -iE 'warn|Fetching|Downloading|it/s\]|B/s\]'
