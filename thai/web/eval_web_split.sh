#!/usr/bin/env bash
# Per-question Mind2Web accuracy for web1 and run 3 on GPU 1 (fits beside the :8011 cascade).
#   nohup bash thai/web/eval_web_split.sh > thai/out/web1_split.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
for m in web1 run3; do
  docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/web -v docker_hf-cache:/hf --shm-size 2g laya-train \
    python eval_web.py --model /work/thai/out/laya-th-$m --out /work/thai/out/${m}_m2w_split.json
done
echo done
