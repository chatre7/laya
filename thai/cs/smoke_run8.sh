#!/usr/bin/env bash
# Smoke-test label_cs8.py on a few records per source (teacher gate, Pantip human items, target rebuild), then start run8.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."
sed -i "s/\r$//" thai/cs/*.py thai/cs/run8.sh
docker run --rm -v "$PWD":/work -w /work/thai/cs -v docker_hf-cache:/hf laya-train python label_cs8.py --from-records cs7 \
  --limit 12 --pantip-repeat 2 --prefix smoke8 --teacher http://172.18.72.145:8010 --workers 4 2>&1 \
  | grep -v "Warning\|it/s\]" | grep -E "reusing|records to label|teacher labelled|\"items\"|\"gated_out\"|\"records\"|pantip|b77|iqa|Traceback|Error|rror:" || true
ls thai/data/cs/ | grep smoke8 || echo "smoke produced no files"
python3 - <<'EOF'
import json
n = 0
for line in open("thai/data/cs/smoke8.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["source"].startswith("pantip") and n < 2:
        n += 1
        print(r["id"], r["labels"], {k: [round(x, 2) for x in v][:6] for k, v in r["targets"].items() if k in ("business", "intent", "department", "urgency")})
    if r["source"] in ("b77", "iqa") and n < 4:
        n += 1
        print(r["id"], r["labels"], "teacher_p", r.get("teacher_p"))
EOF
find thai/data/cs -maxdepth 1 -name "smoke8*" -delete
(nohup bash thai/cs/run8.sh > thai/out/run8.log 2>&1 < /dev/null &)
sleep 5
cat thai/out/run8.log
