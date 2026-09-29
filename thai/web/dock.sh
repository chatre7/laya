#!/usr/bin/env bash
# Run a python script from thai/web inside the laya-train image (CPU; set GPU=1 for GPU 1).
#   bash thai/web/dock.sh inspect_m2w.py [args...]
cd "$(dirname "$0")/../.."
GPUARGS=()
[ "${GPU:-}" = "1" ] && GPUARGS=(--gpus '"device=1"' --shm-size 2g)
docker run --rm "${GPUARGS[@]}" -v "$PWD":/work -w /work/thai/web -v docker_hf-cache:/hf laya-train python "$@" 2>&1 | grep -v "Warning\|it/s\]"
