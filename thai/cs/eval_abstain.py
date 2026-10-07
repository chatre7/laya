"""Abstain as a designed output: on the hand-labelled real sets (360 long posts, 714 short versions, 300 reviews), for each
question and confidence threshold, how often the student answers on its own (coverage), how right it is when it does,
how right it is on the part it would hand over, and what the hand-over is worth: to the teacher (OpenThai-SystemOne at
:8010, as the cascade does today) or to a person (counted as always right - the ceiling). Confidence is the largest
probability of the question.

    python eval_abstain.py --student /work/thai/out/laya-th-run19 --out /work/thai/out/abstain19.json
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import load_agent  # noqa: E402
from cs_questions import SHARED, intent_question  # noqa: E402
from eval_real import load as load_real  # noqa: E402

QS = ("intent", "department", "urgency")
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def questions(biz):
    return {"intent": intent_question(biz), "department": SHARED["department"], "urgency": SHARED["urgency"]}


def read(ans, q):
    """(answer, confidence) in the label's form."""
    a = ans[q]
    if q == "urgency":
        p = a["probabilities"]
        lv = max(range(3), key=lambda i: p[str(i)])
        return lv, p[str(lv)]
    return a["choice"], a["probabilities"][a["choice"]]


def teacher(url, text, qs):
    body = json.dumps({"state": text, "questions": qs}, ensure_ascii=False).encode()
    req = urllib.request.Request(url + "/v1/systemone", body, {"content-type": "application/json"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["answers"]
        except Exception:  # noqa: BLE001
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="/work/thai/out/laya-th-run19")
    ap.add_argument("--teacher", default="http://172.18.72.145:8010")
    ap.add_argument("--long", default="/work/thai/data/domain/real_cs_eval.jsonl")
    ap.add_argument("--short", default="/work/thai/data/cs/pantip_short.jsonl")
    ap.add_argument("--frustration", default="/work/thai/data_domain/frustration_real_eval.json")
    ap.add_argument("--sample", default="/work/thai/data_domain/play_reviews_eval_sample.jsonl")
    ap.add_argument("--labels", default="/work/thai/data_domain/play_labels.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    rows = [{"set": r["set"], "business": r["business"], "text": r["text"], "labels": {k: r["labels"][k] for k in QS}} for r in load_real(args)]
    labels = json.load(open(args.labels, encoding="utf-8"))["labels"]
    for l in open(args.sample, encoding="utf-8"):
        r = json.loads(l)
        if r["id"] in labels:
            it, de, ur = labels[r["id"]]
            rows.append({"set": "reviews", "business": r["business"], "text": r["text"], "labels": {"intent": it, "department": de, "urgency": int(ur)}})
    print(f"{len(rows)} rows: " + ", ".join(f"{s} {sum(r['set'] == s for r in rows)}" for s in ("long", "short", "reviews")), flush=True)

    agent = load_agent(args.student)
    for r in rows:
        r["student"] = agent.predict(r["text"], questions(r["business"]))["answers"]
    del agent
    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(lambda r: teacher(args.teacher, r["text"], questions(r["business"])), rows))
    for r, o in zip(rows, outs):
        r["teacher"] = o
    print(f"teacher answered {sum(o is not None for o in outs)}/{len(rows)}", flush=True)

    report = {}
    for sset in ("long", "short", "reviews", "all"):
        sub = [r for r in rows if sset == "all" or r["set"] == sset]
        report[sset] = {}
        print(f"\n== {sset} ({len(sub)} rows)")
        print(f"{'question':11s} {'thr':>4s} {'answers':>8s} {'right when':>10s} {'right when':>10s} {'teacher on':>10s} {'cascade':>8s} {'to a person':>11s} | student alone / teacher alone")
        print(f"{'':11s} {'':>4s} {'itself':>8s} {'it does':>10s} {'it hands over':>10s} {'handed':>10s} {'(teacher)':>8s} {'(ceiling)':>11s} |")
        for q in QS:
            st = [(read(r["student"], q), r["labels"][q], read(r["teacher"], q)[0] if r["teacher"] else None) for r in sub]
            alone = sum(a == g for (a, _), g, _ in st) / len(st)
            t_alone = sum(t == g for (_, _), g, t in st) / len(st)
            report[sset][q] = {"n": len(st), "student": round(alone, 4), "teacher": round(t_alone, 4), "thresholds": {}}
            for th in THRESHOLDS:
                cov = [(a == g, t == g) for (a, c), g, t in st if c >= th]
                rest = [(a == g, t == g) for (a, c), g, t in st if c < th]
                coverage = len(cov) / len(st)
                acc_cov = sum(x for x, _ in cov) / max(1, len(cov))
                acc_rest = sum(x for x, _ in rest) / max(1, len(rest))
                t_rest = sum(y for _, y in rest) / max(1, len(rest))
                cascade = (sum(x for x, _ in cov) + sum(y for _, y in rest)) / len(st)
                person = (sum(x for x, _ in cov) + len(rest)) / len(st)
                report[sset][q]["thresholds"][str(th)] = {"coverage": round(coverage, 4), "acc_covered": round(acc_cov, 4), "acc_rest_student": round(acc_rest, 4),
                                                        "acc_rest_teacher": round(t_rest, 4), "cascade": round(cascade, 4), "person": round(person, 4)}
                print(f"{q:11s} {th:4.1f} {coverage:8.0%} {acc_cov:10.1%} {acc_rest:10.1%} {t_rest:10.1%} {cascade:8.1%} {person:11.1%} | {alone:.1%} / {t_alone:.1%}")
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
