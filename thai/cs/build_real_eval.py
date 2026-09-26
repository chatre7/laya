"""Turn the hand-labelled Pantip samples (data_domain/labels_<business>.json + pantip_<business>_review_sample.jsonl) into
eval_thai.py-format records (intent with that business's list + other, department, urgency) and score HTTP endpoints on them.

    python build_real_eval.py --out thai/data_domain/real_cs_eval.jsonl --model teacher=http://172.18.72.145:8010 --model student=http://172.18.72.145:8011
"""
import argparse
import json
import os
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cs_questions import SHARED, intent_question  # noqa: E402

DATA = os.path.join(HERE, "..", "data_domain")


def build():
    recs = []
    for biz in ("telecom", "banking", "insurance"):
        lab = json.load(open(os.path.join(DATA, f"labels_{biz}.json"), encoding="utf-8"))["labels"]
        rows = [json.loads(l) for l in open(os.path.join(DATA, f"pantip_{biz}_review_sample.jsonl"), encoding="utf-8")]
        for i, r in enumerate(rows):
            if str(i) not in lab:
                continue
            intent, dept, urg = lab[str(i)]
            recs.append({"id": f"{biz}-{i}", "source": biz, "state": r["text"],
                         "questions": {"intent": intent_question(biz), "department": SHARED["department"], "urgency": SHARED["urgency"]},
                         "labels": {"intent": intent, "department": dept, "urgency": int(urg)}})
    return recs


def ask(url, r):
    body = json.dumps({"state": r["state"], "questions": r["questions"], "order_invariant": False}, ensure_ascii=False).encode()
    req = urllib.request.Request(url + "/v1/systemone", body, {"content-type": "application/json"})
    for _ in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))["answers"]
        except Exception:  # noqa: BLE001
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], help="name=url")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    recs = build()
    with open(args.out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = defaultdict(int)
    for r in recs:
        by[r["source"]] += 1
    n_other = sum(1 for r in recs if r["labels"]["intent"] == "other")
    print(f"{len(recs)} records {dict(by)}; intent=other on {n_other} ({n_other / len(recs):.0%}) -> {args.out}")
    for spec in args.model:
        name, url = spec.split("=", 1)
        with ThreadPoolExecutor(args.workers) as ex:
            answers = list(ex.map(lambda r: ask(url, r), recs))
        st = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0]))  # source -> question -> [n, ok, mae]
        conf = defaultdict(lambda: [0, 0, 0, 0])  # source -> [n_other_label, other_pred_on_other, n_in, other_pred_on_in]
        for r, a in zip(recs, answers):
            if a is None:
                continue
            s = r["source"]
            lab = r["labels"]
            st[s]["intent"][0] += 1
            st[s]["intent"][1] += a["intent"]["choice"] == lab["intent"]
            st[s]["department"][0] += 1
            st[s]["department"][1] += a["department"]["choice"] == lab["department"]
            st[s]["urgency"][0] += 1
            st[s]["urgency"][1] += round(a["urgency"]["score"]) == lab["urgency"]
            st[s]["urgency"][2] += abs(a["urgency"]["score"] - lab["urgency"])
            c = conf[s]
            if lab["intent"] == "other":
                c[0] += 1
                c[1] += a["intent"]["choice"] == "other"
            else:
                c[2] += 1
                c[3] += a["intent"]["choice"] == "other"
        print(f"\n== {name} ({url})")
        print(f"{'source':10s} {'n':>4s} {'intent':>7s} {'dept':>6s} {'urg':>6s} {'urgMAE':>7s} | other recall / false-other")
        tot = [0, 0, 0, 0, 0, 0]
        for s in ("telecom", "banking", "insurance"):
            q = st[s]
            n = q["intent"][0]
            c = conf[s]
            print(f"{s:10s} {n:4d} {q['intent'][1] / n:7.3f} {q['department'][1] / n:6.3f} {q['urgency'][1] / n:6.3f} {q['urgency'][2] / n:7.2f} | "
                  f"{c[1]}/{c[0]} {c[3]}/{c[2]}")
            tot[0] += n
            tot[1] += q["intent"][1]
            tot[2] += q["department"][1]
            tot[3] += q["urgency"][1]
        print(f"{'all':10s} {tot[0]:4d} {tot[1] / tot[0]:7.3f} {tot[2] / tot[0]:6.3f} {tot[3] / tot[0]:6.3f}")


if __name__ == "__main__":
    main()
