"""Evaluate the OpenThai-SystemOne HTTP server on the same eval.jsonl that eval_thai.py uses for laya,
with the same metrics, so the two models can be compared on identical records.

    python laya-ft/eval_ots.py --eval laya-ft/results/eval.jsonl --out laya-ft/results/ots.json
"""
import argparse
import json
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from eval_thai import TICKETS, TICKET_Q  # noqa: E402  (same 5 hand-written cases)


def call(url, state, questions, order_invariant=False):
    body = json.dumps({"state": state, "questions": questions, "order_invariant": order_invariant}, ensure_ascii=False).encode()
    req = urllib.request.Request(f"{url}/v1/systemone", body, {"content-type": "application/json"})
    t = time.perf_counter()
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return r, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://172.18.72.145:8010")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8, help="concurrent requests (the server batches them)")
    ap.add_argument("--order-invariant", action="store_true")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    stats = defaultdict(lambda: {"n": 0, "correct": 0, "abs_err": 0.0, "type": ""})
    lat, errors = [], 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda r: _safe(args.url, r, args.order_invariant), records))
    for r, out in zip(records, results):
        if out is None:
            errors += 1
            continue
        res, ms = out
        lat.append(ms)
        for qid, q in r["questions"].items():
            if qid not in r["labels"]:
                continue
            a, lab = res["answers"][qid], r["labels"][qid]
            s = stats[f'{r["source"]}:{q["type"]}']
            s["type"] = q["type"]
            s["n"] += 1
            if q["type"] == "choice":
                s["correct"] += a["choice"] == lab
            elif q["type"] == "noul":
                s["correct"] += (a["noul"] > 0.5) == bool(lab)
            else:
                s["correct"] += round(a["score"]) == int(lab)
                s["abs_err"] += abs(a["score"] - int(lab))

    tickets = {"department": 0, "refund": 0, "frustration_mae": 0.0, "rows": []}
    for state, dept, fr, refund in TICKETS:
        a = call(args.url, state, TICKET_Q, args.order_invariant)[0]["answers"]
        tickets["department"] += a["department"]["choice"] == dept
        tickets["refund"] += (a["refund"]["noul"] > 0.5) == refund
        tickets["frustration_mae"] += abs(a["frustration"]["score"] - fr) / len(TICKETS)
        tickets["rows"].append({"state": state[:30], "dept": a["department"]["choice"], "p": a["department"]["probabilities"][a["department"]["choice"]],
                                "frustration": round(a["frustration"]["score"], 2), "refund": a["refund"]["noul"]})

    summary = {"model": f"openthai-systemone @ {args.url}", "order_invariant": args.order_invariant, "errors": errors,
               "mean_ms": sum(lat) / max(1, len(lat)), "wall_s": time.perf_counter() - t0, "tickets": tickets, "sources": {}}
    print(f"\n{summary['model']}: {len(lat)} records, {errors} errors, {summary['mean_ms']:.0f} ms/record (8 concurrent), wall {summary['wall_s']:.0f} s")
    print(f"{'source:type':28s} {'n':>5s} {'acc':>6s} {'mae':>6s}")
    for key in sorted(stats):
        s = stats[key]
        acc = s["correct"] / max(1, s["n"])
        mae = s["abs_err"] / max(1, s["n"]) if s["type"] == "score" else None
        summary["sources"][key] = {"n": s["n"], "acc": round(acc, 4), "mae": None if mae is None else round(mae, 4)}
        print(f"{key:28s} {s['n']:5d} {acc:6.3f} {'' if mae is None else f'{mae:6.3f}'}")
    print(f"tickets: department {tickets['department']}/5  refund {tickets['refund']}/5  frustration MAE {tickets['frustration_mae']:.2f}")
    json.dump(summary, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def _safe(url, r, oi):
    try:
        return call(url, r["state"], r["questions"], oi)
    except Exception as e:
        print("error:", r["id"], type(e).__name__, str(e)[:120])
        return None


if __name__ == "__main__":
    main()
