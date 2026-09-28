"""Speed and parity of laya 0.3.20's TileLang fast path on our checkpoint, with real Pantip states and the call-center
question set (intent with the business's list + department + urgency): stock bf16 forward vs `agent.accelerate()`.

    python bench_fast.py --model /work/thai/out/laya-th-run8 --eval /work/thai/data/domain/real_cs_eval.jsonl --out /work/thai/out/bench_fast8.json
"""
import argparse
import json
import statistics
import time

import torch

import laya


def timeit(fn, n):
    lat = []
    for i in range(n):
        t = time.perf_counter()
        fn(i)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": round(statistics.median(lat), 1), "mean_ms": round(statistics.mean(lat), 1), "p95_ms": round(sorted(lat)[int(0.95 * len(lat)) - 1], 1)}


def choices(res):
    return {q: (a.get("choice") if "choice" in a else round(float(a["score"]), 2)) for q, a in res["answers"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval", default="/work/thai/data/domain/real_cs_eval.jsonl")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(args.eval, encoding="utf-8")][: args.n]
    states = [r["state"] for r in recs]
    q_of = [r["questions"] for r in recs]
    agent = laya.Agent(args.model, device="cuda")
    print("laya", laya.__version__, "dtype", agent.dtype, "gpu", torch.cuda.get_device_name(0), flush=True)
    out = {"laya": laya.__version__, "n": len(recs), "batch": args.batch, "gpu": torch.cuda.get_device_name(0)}

    def run_single(i):
        return agent.predict(states[i], q_of[i])

    def run_batch(i):
        j = (i * args.batch) % len(recs)
        return agent.predict_batch(states[j: j + args.batch], q_of[j], batch_size=args.batch)

    for _ in range(5):
        run_single(0)
        run_batch(0)
    stock = [choices(run_single(i)) for i in range(len(recs))]
    out["stock"] = {"single": timeit(run_single, len(recs)), f"batch{args.batch}": timeit(run_batch, 20)}
    print("stock", out["stock"], flush=True)

    t = time.perf_counter()
    try:
        agent.accelerate(strict=True)
        out["accelerate_s"] = round(time.perf_counter() - t, 1)
    except Exception as e:  # noqa: BLE001
        out["fast_error"] = f"{type(e).__name__}: ...{str(e)[-1200:]}"
        print("fast path unavailable:", out["fast_error"], flush=True)
        json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        return
    for _ in range(5):
        run_single(0)
        run_batch(0)
    fast = [choices(run_single(i)) for i in range(len(recs))]
    out["fast"] = {"single": timeit(run_single, len(recs)), f"batch{args.batch}": timeit(run_batch, 20), "warmup_s": out["accelerate_s"]}
    agree = sum(a[q] == b[q] for a, b in zip(stock, fast) for q in a)
    total = sum(len(a) for a in stock)
    out["fast"]["argmax_agreement"] = round(agree / total, 4)
    out["fast"]["disagreements"] = [(recs[i]["id"], q, a[q], b[q]) for i, (a, b) in enumerate(zip(stock, fast)) for q in a if a[q] != b[q]][:20]
    print("fast", out["fast"], flush=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
