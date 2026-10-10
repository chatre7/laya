#!/usr/bin/env bash
# Run 20, step 1: Saluki (Qwen3.8-27B 2-bit, llama.cpp :8013) labels the pool texts that contain a cue word of any new
# question, beside the cascade. Resumes if restarted. ~5-6 h for ~4,000 texts on the A2.
#   nohup bash thai/cs/saluki_label.sh > thai/out/saluki_label.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
bash thai/cs/saluki_up.sh || exit 1
for i in $(seq 1 90); do curl -s -m 3 localhost:8013/health | grep -q '"ok"' && break; sleep 5; done
python3 thai/cs/label_new_gen.py --cue-only --url http://localhost:8013 --model saluki --out thai/data/cs/new_saluki.jsonl --workers 4
docker rm -f llama-saluki >/dev/null 2>&1 || true
echo "== [$(date +%H:%M)] done: $(wc -l < thai/data/cs/new_saluki.jsonl) labelled"
