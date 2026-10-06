"""Print where a stored set of answers to the new questions disagrees with the hand labels, to read before changing the
labeller's rules (or the hand labels, when the labeller is the one that is right).

    python3 thai/cs/show_new_errors.py thai/data/cs/new_gen.eval.jsonl [question]
"""
import json
import sys

QS = ["churn_threat", "external_threat", "third_party", "contact_effort"]
root = "thai"
labels = json.load(open(f"{root}/data_domain/new_labels.json", encoding="utf-8"))["labels"]
only = sys.argv[2] if len(sys.argv) > 2 else ""
for l in open(sys.argv[1], encoding="utf-8"):
    d = json.loads(l)
    if d["id"] not in labels:
        continue
    for k, q in enumerate(QS):
        if only and q != only:
            continue
        a, g = d["answers"][q], labels[d["id"]][k]
        p = max(range(4), key=a.__getitem__) if q == "contact_effort" else int(a > 0.5)
        bad = (p > 0) != (g > 0) if q == "contact_effort" else p != g
        if bad:
            print(f"{q[:8]} hand {g} model {p} | {d['id']} | {d['text'][:170]}")
