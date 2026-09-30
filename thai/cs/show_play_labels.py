"""Print reviews with the student's intent and the teacher's top-2 (label_play.py output), to see where they disagree."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/trbdevsysadmin/laya/thai/data/cs/smoke12_play.jsonl"
for i, l in enumerate(open(path, encoding="utf-8")):
    if i >= 45:
        break
    r = json.loads(l)
    keys = list(json.loads(json.dumps(r["targets"]["intent"])) and r["targets"]["intent"]) if False else None
    t = r["targets"]["intent"]
    from cs_questions import INTENTS  # noqa: E402
    names = list(INTENTS[r["business"]]) + ["other"]
    top = sorted(range(len(t)), key=t.__getitem__, reverse=True)[:2]
    print(f"{r['business'][:4]} | {r['text'][:70]:70s} | S: {r['student_intent']:28s} | T: " + " ".join(f"{names[j]}={t[j]:.2f}" for j in top))
