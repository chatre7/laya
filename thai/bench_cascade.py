"""Benchmark + equivalence check for the cascade server's student batching.

1. Equivalence: the same N eval records sent one at a time and then at high concurrency must give the same answers
   (max abs probability difference reported; batching pads sequences, bf16 noise is expected below ~0.01).
2. Latency/throughput at several concurrency levels on the student-only path (questions the student keeps), and on
   the full ticket (one question goes to the teacher).

    python thai/bench_cascade.py --url http://172.18.72.145:8011 --eval thai/data_eval_sample.jsonl
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TICKET = {
    "state": "โดนหักเงินซ้ำสองครั้งเมื่อวานนี้ ขอเงินคืนด่วนนะครับ",
    "questions": {
        "department": {"type": "choice", "instructions": "ทีมใดควรรับผิดชอบ",
                       "criteria": {"billing": "ค่าบริการ ใบแจ้งหนี้", "technical": "ระบบใช้งานไม่ได้", "sales": "สมัคร เปลี่ยนแพ็กเกจ"}},
        "frustration": {"type": "score", "instructions": "ลูกค้าไม่พอใจแค่ไหน", "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]},
        "refund": {"type": "noul", "instructions": "ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"},
    },
}
from smoke_cascade import WIDE as STUDENT_ONLY  # 60-intent question the run 4 student keeps with p ~1.0: measures the student path alone


def post(url, body, timeout=120):
    req = urllib.request.Request(url + "/v1/systemone", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    t = time.perf_counter()
    res = json.load(urllib.request.urlopen(req, timeout=timeout))
    return res, (time.perf_counter() - t) * 1000


def probs(ans):
    out = []
    for q in sorted(ans):
        a = ans[q]
        out += list(a["probabilities"].values()) if "probabilities" in a else [a["noul"]]
    return out


def run(url, body, n, workers):
    with ThreadPoolExecutor(workers) as ex:
        t = time.perf_counter()
        lat = sorted(ex.map(lambda _: post(url, body)[1], range(n)))
        wall = time.perf_counter() - t
    return {"n": n, "workers": workers, "mean": statistics.mean(lat), "p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95)], "rps": n / wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://172.18.72.145:8011")
    ap.add_argument("--eval", default="", help="jsonl with state/questions; first --n records are used for the equivalence check")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--skip-teacher", action="store_true")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    print("stats before:", {k: v for k, v in json.load(urllib.request.urlopen(url + "/stats")).items() if k.startswith("student_")})

    if args.eval:
        rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")][: args.n]
        bodies = [{"state": r["state"], "questions": r["questions"]} for r in rows]
        single = [post(url, b)[0]["answers"] for b in bodies]
        with ThreadPoolExecutor(32) as ex:
            batched = [r[0]["answers"] for r in ex.map(lambda b: post(url, b), bodies)]
        diffs = [abs(x - y) for s, b in zip(single, batched) for x, y in zip(probs(s), probs(b))]
        same = sum(1 for s, b in zip(single, batched) if all(s[q].get("choice") == b[q].get("choice") for q in s))
        print(f"equivalence on {len(bodies)} records: max |p_single - p_batched| = {max(diffs):.4f}, mean {statistics.mean(diffs):.5f}, same argmax {same}/{len(bodies)}")

    print(f"\n{'path':14s} {'workers':>7s} {'mean ms':>8s} {'p50':>6s} {'p95':>6s} {'req/s':>7s}")
    for name, body in (("student-only", STUDENT_ONLY),) + (() if args.skip_teacher else (("ticket", TICKET),)):
        post(url, body)  # warm
        for w in (1, 8, 32):
            r = run(url, body, max(32, w * 8), w)
            print(f"{name:14s} {w:7d} {r['mean']:8.0f} {r['p50']:6.0f} {r['p95']:6.0f} {r['rps']:7.1f}")
    print("\nstats after:", {k: v for k, v in json.load(urllib.request.urlopen(url + "/stats")).items() if k.startswith("student_")})


if __name__ == "__main__":
    main()
