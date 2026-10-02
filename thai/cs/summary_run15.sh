#!/usr/bin/env bash
# Print the run 15 numbers from the log and result files.
cd "$(dirname "$0")/../out"
echo "--- items"; grep -E 'composed:|mined |"new_items"|"replay"|"items"' run15.log
echo "--- mood: stars, ambiguous set, held-out composed sets (run 14 then run 15)"
grep -E '^== laya-th-run1[45]: reviews|^  [1-5]  |1-2 stars called|frustration separates|ambiguous / sarcastic|held-out composed' run15.log
echo "--- real sets, run 15"; awk '/^== laya-th-run15$/{f=1} f&&/^(long|short)/{print}' run15.log
echo "--- play reviews"; grep -E '^(telecom|banking|insurance|all) +n=' run15.log
echo "--- cs9 held-out / public"
python3 - <<'EOF'
import json
for n in ("run14_cs9", "run15_cs9", "run14", "run15"):
    try:
        d = json.load(open(n + ".json"))
        ws = d["sources"].get("wisesight:choice", {}).get("acc")
        print(n, d["overall"], "wisesight", ws)
    except Exception as e:
        print(n, "missing", type(e).__name__)
EOF
