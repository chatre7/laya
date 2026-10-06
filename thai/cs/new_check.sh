#!/usr/bin/env bash
# The hand check of the new-question teacher: the pilot's stored answers (asked four at a time, in batches), then the
# teacher asked again per text - four together and one question per request - and run 16, on data_domain/new_labels.json.
#   bash thai/cs/new_check.sh > thai/out/new_check.log 2>&1
set -uo pipefail
cd "$(dirname "$0")/../.."
docker run --rm --gpus '"device=1"' -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf --shm-size 2g -e PYTHONUNBUFFERED=1 laya-train-clef \
  python eval_new.py --teacher /work/thai/data/cs/new_pilot.jsonl --models "${1:-Cloudflare/clef-flash,/work/thai/out/laya-th-run16}" --separate \
  --out /work/thai/out/new_check_pilot.json 2>&1 | grep --line-buffered -E '^==|^  [a-z_]+ |hand-labelled|Error|Traceback'
echo "== [$(date +%H:%M)] done"
