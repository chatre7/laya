#!/usr/bin/env bash
# Print the run 16 numbers from the log and result files (run 15 beside them where a file exists).
cd "$(dirname "$0")/../out"
echo "--- items"; grep -E 'composed:|mined |wisesight |"new_items"|"replay"|"items"' run16.log
echo "--- mood: stars, ambiguous set, held-out composed sets (run 14 then run 16)"
grep -E '^== laya-th-run1[46]: reviews|^  [1-5]  |1-2 stars called|frustration separates|ambiguous / sarcastic|gold -> answer|held-out composed' run16.log
echo "--- real sets, run 16"; awk '/^== laya-th-run16$/{f=1} f&&/^(long|short)/{print}' run16.log
echo "--- play reviews"; grep -E '^(telecom|banking|insurance|all) +n=' run16.log
echo "--- cs9 held-out / public"
python3 - <<'EOF'
import json
for n in ("run14_cs9", "run15_cs9", "run16_cs9", "run14", "run15", "run16"):
    try:
        d = json.load(open(n + ".json"))
        ws = d["sources"].get("wisesight:choice", {}).get("acc")
        print(n, d["overall"], "wisesight", ws)
    except Exception as e:
        print(n, "missing", type(e).__name__)
EOF
