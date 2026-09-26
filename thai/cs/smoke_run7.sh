#!/usr/bin/env bash
# Smoke-test label_cs.py --from-records on a handful of generated out-of-scope texts, then start the run 7 watcher.
set -euo pipefail
cd "$(dirname "$0")/../.."
sed -i "s/\r$//" thai/cs/label_cs.py thai/cs/run7.sh
mkdir -p thai/data/cs/_smoke
cp /tmp/other_sample.jsonl thai/data/cs/_smoke/other_sample.jsonl
docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train python label_cs.py --from-records cs6 \
  --other-file /work/thai/data/cs/_smoke/other_sample.jsonl --out /work/thai/data/cs --prefix smoke7 \
  --teacher http://172.18.72.145:8010 --workers 4 2>&1 | grep -v "Warning\|it/s\]" | grep -E "reusing|records to label|teacher labelled|\"items\"|other_gen|Traceback|Error" || true
ls thai/data/cs/ | grep smoke7 || echo "smoke produced no files"
find thai/data/cs -maxdepth 1 -name "smoke7*" -delete
(nohup bash thai/cs/run7.sh > thai/out/run7.log 2>&1 < /dev/null &)
sleep 3
cat thai/out/run7.log
grep -v "Warning\|it/s\]" thai/data/cs/gen_other.log | tail -2
