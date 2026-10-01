"""How good are labels on which the LLM and the student agree? On the 360 hand-labelled Pantip posts
(label_pantip_llm.py --eval -> pantip_llm_eval.jsonl): agreement rate per question and accuracy on the agreed rows.

    python agree_check.py --student /work/thai/out/laya-th-run11
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402
from cs_questions import SHARED, intent_question  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/work/thai/data/cs/pantip_llm_eval.jsonl")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run11")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    agent = laya.Agent(args.student, device="cuda")
    c = Counter()
    for r in rows:
        a = agent.predict(r["text"], {"intent": intent_question(r["business"]), "department": SHARED["department"], "urgency": SHARED["urgency"]})["answers"]
        s = [a["intent"]["choice"], a["department"]["choice"], round(a["urgency"]["score"])]
        llm = r["llm"] or ["other", "support", 0]
        g = [r["gold"]["intent"], r["gold"]["department"], r["gold"]["urgency"]]
        for i, q in enumerate(("intent", "department", "urgency")):
            c[q, "n"] += 1
            c[q, "student"] += s[i] == g[i]
            c[q, "llm"] += llm[i] == g[i]
            if s[i] == llm[i]:
                c[q, "agree"] += 1
                c[q, "agree_ok"] += s[i] == g[i]
            c[q, "either"] += s[i] == g[i] or llm[i] == g[i]
    for q in ("intent", "department", "urgency"):
        n = c[q, "n"]
        print(f"{q:10s} student {c[q, 'student'] / n:.3f}  llm {c[q, 'llm'] / n:.3f}  agree on {c[q, 'agree']}/{n} = {c[q, 'agree'] / n:.2f}, "
              f"right when they agree {c[q, 'agree_ok'] / max(1, c[q, 'agree']):.3f}; at least one right {c[q, 'either'] / n:.3f}")


if __name__ == "__main__":
    main()
