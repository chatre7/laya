#!/usr/bin/env bash
# Cascade sweep for the run 4 student: on the held-out call-center set and on the public set, option gate off.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=(docker run --rm --gpus "\"device=1\"" -v "$PWD":/work -w /work/thai -v docker_hf-cache:/hf --shm-size 2g laya-train)
echo "== [$(date +%H:%M)] cascade run4 on cc_eval_human (option gate off)"
"${RUN[@]}" python cascade.py --student /work/thai/out/laya-th-run4 --eval /work/thai/data/cc/cc_eval_human.jsonl --out /work/thai/out/cascade4_cc.json --max-options 60
echo "== [$(date +%H:%M)] cascade run4 on the public eval set (option gate off)"
"${RUN[@]}" python cascade.py --student /work/thai/out/laya-th-run4 --eval /work/thai/data/eval.jsonl --out /work/thai/out/cascade4.json --max-options 60
"${RUN[@]}" chmod -R a+rX /work/thai/out
echo "== [$(date +%H:%M)] done"
