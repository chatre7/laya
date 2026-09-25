"""Teacher gate for the customer-service rewrites: OpenThai-SystemOne answers the per-business intent question (+ `business`)
on every Thai rewrite; report agreement with the source intent per business/style and keep p(source intent) for filtering.

    python check_cs_rewrites.py --inp data/cs/cs_colloquial.jsonl --out data/cs/cs_colloquial_checked.jsonl
"""
import argparse
import collections
import json
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs_questions import BUSINESS_Q, intent_question  # noqa: E402


def ask(url, r):
    qs = {"business": BUSINESS_Q, "intent": intent_question(r["business"])}
    body = json.dumps({"state": r["text"], "questions": qs, "order_invariant": False}, ensure_ascii=False).encode()
    req = urllib.request.Request(url + "/v1/systemone", body, {"content-type": "application/json"})
    for _ in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))["answers"]
        except Exception:  # noqa: BLE001
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="http://localhost:8010")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    done = [0]

    def work(r):
        a = ask(args.teacher, r)
        done[0] += 1
        if done[0] % 5000 == 0:
            print(f"  {done[0]}/{len(rows)}", flush=True)
        return a

    with ThreadPoolExecutor(args.workers) as ex:
        answers = list(ex.map(work, rows))
    st = collections.defaultdict(lambda: [0, 0, 0.0, 0])
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for r, a in zip(rows, answers):
            if a is None:
                continue
            r["teacher_intent"] = a["intent"]["choice"]
            r["teacher_p"] = a["intent"]["probabilities"].get(r["intent"], 0.0)
            r["teacher_business"] = a["business"]["choice"]
            r["business_p"] = a["business"]["probabilities"].get(r["business"], 0.0)
            ok = r["teacher_intent"] == r["intent"]
            for key in (r["business"], r["business"] + " / " + r["style"]):
                s = st[key]
                s[0] += 1
                s[1] += ok
                s[2] += r["teacher_p"]
                s[3] += r["teacher_business"] == r["business"]
            kept += r["teacher_p"] >= 0.5
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{'business / style':40s} {'n':>6s} {'intent agree':>12s} {'p(intent)':>10s} {'business ok':>12s}")
    for key, s in sorted(st.items()):
        print(f"{key:40s} {s[0]:6d} {s[1] / s[0]:12.3f} {s[2] / s[0]:10.3f} {s[3] / s[0]:12.3f}")
    print(f"kept at p(intent) >= 0.5: {kept}/{len(rows)} ({kept / max(1, len(rows)):.0%})")


if __name__ == "__main__":
    main()
