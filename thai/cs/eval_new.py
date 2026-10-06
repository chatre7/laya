"""Score answers to the new questions (cs_questions.EXTRA) against the hand labels of the check sample
(data_domain/new_labels.json). Two kinds of answerer: the teacher's stored answers (--teacher, the jsonl of
label_new_llm.py) and checkpoints (--models). The sample was drawn by the teacher's answers, so the numbers describe that
sample, not the stream: precision of "yes" at several thresholds, how many of the hand "yes" are found, and for the score
question exact / within one level / "any earlier contact" (level > 0).

    python eval_new.py --teacher /work/thai/data/cs/new_pilot.jsonl --models /work/thai/out/laya-th-run16
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs_questions import EXTRA  # noqa: E402

QS = ["churn_threat", "external_threat", "third_party", "contact_effort"]


def report(name, answers, rows, labels, show):
    """answers: {id: {q: p(yes) | [level probabilities]}}"""
    out = {}
    print(f"== {name}")
    for qi, q in enumerate(QS):
        gold = {r["id"]: labels[r["id"]][qi] for r in rows}
        if EXTRA[q]["type"] == "noul":
            pos = [i for i, g in gold.items() if g]
            line = []
            for th in (0.5, 0.7, 0.9):
                said = [i for i in gold if answers[i][q] > th]
                hit = sum(gold[i] for i in said)
                line.append(f">{th}: {hit}/{len(said)} right, finds {sum(answers[i][q] > th for i in pos)}/{len(pos)}")
                out[f"{q}@{th}"] = {"said": len(said), "right": hit, "found": sum(answers[i][q] > th for i in pos), "yes": len(pos)}
            print(f"  {q:16s} " + " | ".join(line))
            if show:
                texts = {r["id"]: r["text"] for r in rows}
                for i in sorted(gold, key=lambda i: -answers[i][q])[:show]:
                    print(f"      {answers[i][q]:.2f} gold {gold[i]} | {texts[i][:110]}")
        else:
            top = {i: max(range(len(answers[i][q])), key=answers[i][q].__getitem__) for i in gold}
            n = len(gold)
            exact = sum(top[i] == gold[i] for i in gold)
            near = sum(abs(top[i] - gold[i]) <= 1 for i in gold)
            anyc = sum((top[i] > 0) == (gold[i] > 0) for i in gold)
            said = [i for i in gold if top[i] > 0]
            had = [i for i in gold if gold[i] > 0]
            print(f"  {q:16s} exact {exact}/{n}  within one {near}/{n}  any-contact {anyc}/{n}; says contact {len(said)}, right {sum(gold[i] > 0 for i in said)}, "
                  f"finds {sum(top[i] > 0 for i in had)}/{len(had)}")
            out[q] = {"n": n, "exact": exact, "within_one": near, "any_contact": anyc, "said": len(said), "said_right": sum(gold[i] > 0 for i in said),
                      "found": sum(top[i] > 0 for i in had), "had": len(had)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="/work/thai/data_domain/new_check_all.jsonl")
    ap.add_argument("--labels", default="/work/thai/data_domain/new_labels.json")
    ap.add_argument("--teacher", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--show", type=int, default=0, help="print the texts each answerer is surest are yes")
    ap.add_argument("--out", default="")
    ap.add_argument("--separate", action="store_true", help="also ask each question in a request of its own")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.sample, encoding="utf-8")]
    labels = json.load(open(args.labels, encoding="utf-8"))["labels"]
    rows = [r for r in rows if r["id"] in labels]
    print(f"{len(rows)} hand-labelled texts; yes per question {[sum(labels[r['id']][k] > 0 for r in rows) for k in range(4)]}")
    rep = {}
    if args.teacher:
        t = {}
        for p in args.teacher.split(","):
            for l in open(p, encoding="utf-8"):
                d = json.loads(l)
                t[d["id"]] = d["answers"]
        have = [r for r in rows if r["id"] in t]
        rep["teacher"] = report(f"teacher ({len(have)} of the sample)", t, have, labels, args.show)
    for m in [m for m in args.models.split(",") if m]:
        from agents import load_agent
        agent = load_agent(m)
        for mode in ["together"] + (["one at a time"] if args.separate else []):
            ans = {}
            for r in rows:
                if mode == "together":
                    a = agent.predict(r["text"], EXTRA)["answers"]
                else:  # a model that scores questions jointly may answer a question differently in company
                    a = {q: agent.predict(r["text"], {q: EXTRA[q]})["answers"][q] for q in EXTRA}
                ans[r["id"]] = {q: (a[q]["noul"] if EXTRA[q]["type"] == "noul" else [a[q]["probabilities"][str(k)] for k in range(len(EXTRA[q]["criteria"]))])
                                for q in EXTRA}
            name = os.path.basename(m) + ("" if mode == "together" else ", one question per request")
            rep[name] = report(name, ans, rows, labels, args.show)
        del agent
    if args.out:
        json.dump(rep, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
