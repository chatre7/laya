#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=(docker run --rm --gpus "\"device=1\"" -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf --shm-size 2g laya-train)
echo "== [$(date +%H:%M)] cascade run3, max-options 10"
"${RUN[@]}" python cascade.py --student /work/thai/out/laya-th-run3 --out /work/thai/out/cascade3.json --max-options 10
echo "== [$(date +%H:%M)] cascade run3, max-options 60"
"${RUN[@]}" python cascade.py --student /work/thai/out/laya-th-run3 --out /work/thai/out/cascade3_opt60.json --max-options 60
"${RUN[@]}" chmod -R a+rX /work/thai/out
echo "== [$(date +%H:%M)] done"
