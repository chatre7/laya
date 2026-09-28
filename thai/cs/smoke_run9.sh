#!/usr/bin/env bash
# Smoke-test label_cs8.py with two Pantip batches on a few records per source, then start run9.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
sed -i "s/\r$//" thai/cs/*.py thai/cs/run9.sh
docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train python label_cs8.py --from-records cs7 \
  --pantip-sets extra,extra2 --limit 12 --pantip-repeat 2 --prefix smoke9 --teacher http://172.18.72.145:8010 --workers 4 2>&1 \
  | grep -v "Warning\|it/s\]" | grep -E "reusing|records to label|teacher labelled|\"items\"|\"records\"|pantip|Traceback|Error|rror:" || true
ls thai/data/cs/ | grep smoke9 || echo "smoke produced no files"
grep -c '"pantip-telecom-extra2-' thai/data/cs/smoke9.jsonl thai/data/cs/smoke9_eval.jsonl || true
find thai/data/cs -maxdepth 1 -name "smoke9*" -delete
(nohup bash thai/cs/run9.sh > thai/out/run9.log 2>&1 < /dev/null &)
sleep 5
cat thai/out/run9.log
