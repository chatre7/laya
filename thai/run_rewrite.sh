#!/usr/bin/env bash
# Full colloquial rewrite of the Thai customer-support set + teacher quality gate. Unattended.
#   nohup bash thai/run_rewrite.sh > thai/data/cc/run_rewrite.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
echo "== [$(date +%H:%M)] 1/3 rewrite all rows x3 with qwen3-4b on :8012"
python3 rewrite_colloquial.py --inp data/cc/porameht.jsonl --out data/cc/porameht_colloquial.jsonl --n 0 --variants 3 --workers 32 --seed 42
echo "== [$(date +%H:%M)] 2/3 teacher check on :8010"
python3 check_rewrites.py --inp data/cc/porameht_colloquial.jsonl --out data/cc/porameht_colloquial_checked.jsonl --teacher http://localhost:8010 --workers 8
echo "== [$(date +%H:%M)] 3/3 keep p(label) >= 0.5 and customer >= 0.5"
python3 - <<"PY"
import json
rows=[json.loads(l) for l in open("data/cc/porameht_colloquial_checked.jsonl",encoding="utf-8")]
keep=[r for r in rows if r.get("teacher_p",0)>=0.5 and r.get("is_customer",0)>=0.5]
with open("data/cc/porameht_colloquial_kept.jsonl","w",encoding="utf-8") as f:
    for r in keep: f.write(json.dumps({"text":r["text"],"intent":r["intent"],"category":r["category"],"style":r["style"],"src":r["src"]},ensure_ascii=False)+"\n")
print(f"kept {len(keep)}/{len(rows)} ({len(keep)/max(1,len(rows)):.0%})")
PY
echo "== [$(date +%H:%M)] done"
