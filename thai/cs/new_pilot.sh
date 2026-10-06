#!/usr/bin/env bash
# Pilot of run 19: Clef-Flash answers the four new questions on 1,600 pool texts, beside the cascade. The answers are the
# material for the hand check (sample_new_check.py) that decides which questions go into training.
#   nohup bash thai/cs/new_pilot.sh > thai/out/new_pilot.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g -e PYTHONUNBUFFERED=1 laya-train-clef \
  python label_new_llm.py --limit "${1:-1600}" --out /work/thai/data/cs/new_pilot.jsonl 2>&1 | grep --line-buffered -v -iE 'warn|Fetching|it/s\]|B/s\]|^\[transformers\]'
docker run --rm -v "$PWD":/work laya-train chmod -R a+rwX /work/thai/data/cs
echo "== [$(date +%H:%M)] done"
