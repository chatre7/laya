#!/usr/bin/env bash
# Build the Clef image, download Clef-Flash (19 GB) and run the smoke test 4-bit beside the cascade; if the card is too
# small with the cascade up, the log says so and nothing else is touched.
#   nohup bash thai/d2/clef_smoke.sh > thai/out/clef_smoke.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
echo "== [$(date +%H:%M)] build image"
docker build -q -t laya-train-clef -f thai/d2/Dockerfile.clef thai/d2 || { echo "build failed"; exit 1; }
echo "== [$(date +%H:%M)] smoke"
docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/d2 -v docker_hf-cache:/hf --shm-size 2g -e PYTHONUNBUFFERED=1 laya-train-clef \
  python smoke.py Cloudflare/clef-flash 2>&1 | grep -v -iE 'warn|Fetching|it/s\]|B/s\]'
echo "== [$(date +%H:%M)] done"
