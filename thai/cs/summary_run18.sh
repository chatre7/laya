#!/usr/bin/env bash
# Print the run 18 numbers: the laya-thai-decisions suites (run 16 from decisions.log, run 18 from run18.log), then ours.
cd "$(dirname "$0")/../out"
echo "--- items"; grep -E '"new_items"|"replay"|"items"|in our eval sets' run18.log
echo "--- laya-thai-decisions suites: run 16"; awk '/^== laya-th-run16/{f=1;next} /^== /{f=0} f' decisions.log
echo "--- laya-thai-decisions suites: run 18"; awk '/^== laya-th-run18$/{f=1;next} /^== /{f=0} f&&/cases/' run18.log
echo "--- mood: stars, ambiguous set, held-out composed sets (run 16 then run 18)"
grep -E '^== laya-th-run1[68]: reviews|^  [1-5]  |1-2 stars called|frustration separates|ambiguous / sarcastic|held-out composed' run18.log
echo "--- real sets, run 18"; awk '/^== laya-th-run18$/{f=1} f&&/^(long|short)/{print}' run18.log
echo "--- play reviews"; grep -E '^(telecom|banking|insurance|all) +n=' run18.log
echo "--- cs9 held-out / public"
python3 - <<'EOF'
import json
for n in ("run16_cs9", "run18_cs9", "run16", "run18"):
    try:
        d = json.load(open(n + ".json"))
        ws = d["sources"].get("wisesight:choice", {}).get("acc")
        print(n, d["overall"], "wisesight", ws)
    except Exception as e:
        print(n, "missing", type(e).__name__)
EOF
