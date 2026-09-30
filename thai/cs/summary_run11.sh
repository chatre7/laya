#!/usr/bin/env bash
# Print the run 11 numbers from the logs and result files.
cd "$(dirname "$0")/../out"
echo "--- data"; grep -E 'in-scope rows|long Pantip records' run11.log; grep -E '"items"|"real_items"|"replay"' run11.log
echo "--- real sets, run 11"; awk '/^== laya-th-run11/{f=1} f&&/^(long|short)/{print} /^== \[..:..\] 6\/6/{f=0}' run11.log
echo "--- next best action with run 11 intent (rule)"; grep -E 'rule\(' run11.log
echo "--- cs9 held-out / public"
python3 - <<'EOF'
import json
for n in ("run8_cs9", "run10_cs9", "run11_cs9", "run8", "run10", "run11"):
    try:
        d = json.load(open(n + ".json"))
        print(n, d["overall"])
    except Exception as e:
        print(n, "missing", type(e).__name__)
EOF
docker ps --format '{{.Names}} {{.Status}}' | head -4
