#!/usr/bin/env bash
# Print the run 12 numbers from the log and result files.
cd "$(dirname "$0")/../out"
echo "--- labels"; grep -E 'reviews \{|agreement|"agreed"|"play_items"|"items"|"replay"' run14.log
echo "--- real sets, run 12"; awk '/^== laya-th-run14/{f=1} f&&/^(long|short)/{print} /^== laya-th-run11|^==.*play|^== \[..:..\] [34]\//{f=0}' run14.log
echo "--- play reviews (hand labels)"; awk '/hand-labelled reviews/{f=1} f{print} /^== \[/{f=0}' run14.log | grep -vE '^== \['
echo "--- next best action rule"; grep -E 'rule\(' run14.log | grep -v errors
echo "--- cs9 held-out / public"
python3 - <<'EOF'
import json
for n in ("run11_cs9", "run14_cs9", "run11", "run14"):
    try:
        print(n, json.load(open(n + ".json"))["overall"])
    except Exception as e:
        print(n, "missing", type(e).__name__)
EOF
