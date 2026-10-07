"""Does any confidence signal of the student separate its right answers from its wrong ones? eval_abstain.py showed that
the largest probability sits between 0.8 and 0.9 for almost every answer, so a threshold either hands over nothing or
everything. Here, on the same 1,374 hand-labelled rows: for each question and several signals (largest probability, gap
between the top two, entropy, for intent also the largest merged-group probability), the AUROC of "signal is higher when
the answer is right", and what handing over the least confident 10% / 20% / 30% would buy if a person took them.

    python eval_abstain2.py --student /work/thai/out/laya-th-run19 --out /work/thai/out/abstain19b.json
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import load_agent  # noqa: E402
from cs_questions import SHARED, intent_question  # noqa: E402
from eval_real import load as load_real  # noqa: E402
from intent_groups import group_probs  # noqa: E402

QS = ("intent", "department", "urgency")


def auroc(pos, neg):
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank_sum, i = 0.0, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / max(1, len(pos) * len(neg))


def signals(biz, q, a):
    p = a["probabilities"]
    vals = sorted(p.values(), reverse=True)
    ent = -sum(v * math.log(v + 1e-12) for v in vals)
    out = {"max": vals[0], "margin": vals[0] - (vals[1] if len(vals) > 1 else 0.0), "neg_entropy": -ent}
    if q == "intent":
        out["group_max"] = max(group_probs(biz, p).values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="/work/thai/out/laya-th-run19")
    ap.add_argument("--long", default="/work/thai/data/domain/real_cs_eval.jsonl")
    ap.add_argument("--short", default="/work/thai/data/cs/pantip_short.jsonl")
    ap.add_argument("--frustration", default="/work/thai/data_domain/frustration_real_eval.json")
    ap.add_argument("--sample", default="/work/thai/data_domain/play_reviews_eval_sample.jsonl")
    ap.add_argument("--labels", default="/work/thai/data_domain/play_labels.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = [{"set": r["set"], "business": r["business"], "text": r["text"], "labels": {k: r["labels"][k] for k in QS}} for r in load_real(args)]
    labels = json.load(open(args.labels, encoding="utf-8"))["labels"]
    for l in open(args.sample, encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in labels:
            it, de, ur = labels[r["id"]]
            rows.append({"set": "reviews", "business": r["business"], "text": r["text"], "labels": {"intent": it, "department": de, "urgency": int(ur)}})
    agent = load_agent(args.student)
    for r in rows:
        r["ans"] = agent.predict(r["text"], {"intent": intent_question(r["business"]), "department": SHARED["department"], "urgency": SHARED["urgency"]})["answers"]
    report = {}
    print(f"{len(rows)} rows. AUROC = chance that a right answer carries a higher signal than a wrong one (0.5 = useless)")
    for q in QS:
        recs = []
        for r in rows:
            a = r["ans"][q]
            if q == "urgency":
                p = a["probabilities"]
                pred = max(range(3), key=lambda i: p[str(i)])
            else:
                pred = a["choice"]
            recs.append((signals(r["business"], q, a), pred == r["labels"][q]))
        acc = sum(ok for _, ok in recs) / len(recs)
        report[q] = {"n": len(recs), "accuracy": round(acc, 4), "signals": {}}
        print(f"\n== {q}: accuracy {acc:.1%}, {sum(ok for _, ok in recs)} right / {sum(not ok for _, ok in recs)} wrong")
        print(f"{'signal':12s} {'AUROC':>6s} {'range':>14s} | hand over the least confident 10% / 20% / 30% -> right among kept, and among handed")
        for name in recs[0][0]:
            pos = [s[name] for s, ok in recs if ok]
            neg = [s[name] for s, ok in recs if not ok]
            au = auroc(pos, neg)
            order = sorted(recs, key=lambda x: x[0][name])
            cells = []
            for frac in (0.1, 0.2, 0.3):
                k = int(len(order) * frac)
                handed, kept = order[:k], order[k:]
                cells.append(f"{sum(ok for _, ok in kept) / len(kept):.1%} / {sum(ok for _, ok in handed) / max(1, len(handed)):.1%}")
            lo, hi = min(min(pos), min(neg)), max(max(pos), max(neg))
            report[q]["signals"][name] = {"auroc": round(au, 4), "min": round(lo, 4), "max": round(hi, 4), "hand_over": cells}
            print(f"{name:12s} {au:6.3f} {lo:6.3f}..{hi:<6.3f} | " + "   ".join(cells))
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
