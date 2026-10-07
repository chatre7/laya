#!/usr/bin/env bash
# Push run folders to the private HF repo from the training image (the host python has no huggingface_hub). The token
# is read from ~/.cache/huggingface/token on the box, put there by the owner:
#   docker run --rm -it -v ~/.cache/huggingface:/root/.cache/huggingface laya-train hf auth login
#   bash thai/hf_push.sh 16,19
set -euo pipefail
cd "$(dirname "$0")/.."
TOKEN=$(cat ~/.cache/huggingface/token 2>/dev/null || true)
[ -n "$TOKEN" ] || { echo "no token in ~/.cache/huggingface/token"; exit 1; }
docker run --rm -v "$PWD":/work -w /work/thai -e HF_TOKEN="$TOKEN" -e HF_HUB_ENABLE_HF_TRANSFER=0 laya-train \
  python hf_push.py --repo "${REPO:-chatre7/laya-th}" --runs "${1:-16,19}"
