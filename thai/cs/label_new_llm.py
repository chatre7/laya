"""A larger decision model as the labelling teacher for questions laya has never been asked (cs_questions.EXTRA: churn
threat, threat to go outside, writing for someone else, contact effort). Clef-Flash (Cloudflare, 9B, run 4-bit) answers all
four in one pass per text over our real Thai service text, and its probabilities are kept as soft targets for run 19.

The pool: Google Play reviews, Pantip posts (with and without an intent label), their short chat versions (set "train") and
the wisesight posts about the three businesses - minus everything in an eval set (the 300 labelled reviews, the 360
labelled posts and their short versions) and minus texts under 15 characters.

    python label_new_llm.py --limit 1600 --out /work/thai/data/cs/new_pilot.jsonl      # pilot, for the hand check
    python label_new_llm.py --out /work/thai/data/cs/new_llm.jsonl                     # everything; resumes
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs_questions import EXTRA  # noqa: E402

BIZ = ("telecom", "banking", "insurance")
ROOT = "/work/thai"


def key_of(t):
    return " ".join(t.split())[:60]


def pool(root=ROOT, min_chars=15):
    """Every text we may train a new question on: [{id, source, business, text}], eval texts left out."""
    seen = {key_of(json.loads(l)["state"]) for l in open(f"{root}/data/domain/real_cs_eval.jsonl", encoding="utf-8")}
    held = {json.loads(l)["id"] for l in open(f"{root}/data_domain/play_reviews_eval_sample.jsonl", encoding="utf-8")}
    rows = []

    def add(rid, source, biz, text):
        text = " ".join(str(text).split())
        k = key_of(text)
        if len(text) >= min_chars and k not in seen:
            seen.add(k)
            rows.append({"id": rid, "source": source, "business": biz, "text": text})

    for biz in BIZ:
        files = [f"{root}/data/domain/pantip_{biz}.jsonl"] + sorted(glob.glob(f"{root}/data/domain/pantip_{biz}_extra*.jsonl"))
        n = 0
        for p in files:
            if os.path.exists(p):
                for l in open(p, encoding="utf-8"):
                    if l.strip():
                        add(f"pantip-{biz}-{n}", "pantip", biz, json.loads(l).get("text", ""))
                        n += 1
        p = f"{root}/data/domain/wisesight_{biz}.jsonl"
        if os.path.exists(p):
            for i, l in enumerate(open(p, encoding="utf-8")):
                add(f"ws-{biz}-{i}", "wisesight", biz, json.loads(l).get("text", ""))
    for l in open(f"{root}/data/cs/pantip_short.jsonl", encoding="utf-8"):
        s = json.loads(l)
        if s["set"] == "train":
            add(f"short-{s['id']}-{s['variant']}", "short", s["business"], s["text"])
    for l in open(f"{root}/data_domain/play_reviews.jsonl", encoding="utf-8"):
        r = json.loads(l)
        if r["id"] not in held:
            add(r["id"], "play", r["business"], r["text"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Cloudflare/clef-flash")
    ap.add_argument("--out", default=f"{ROOT}/data/cs/new_llm.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="a random sample of this size, the same share of every source")
    ap.add_argument("--max-chars", type=int, default=1500, help="about what the student reads of a long post")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = pool()
    print(f"pool: {len(rows)} texts {dict(Counter(r['source'] for r in rows))}", flush=True)
    if args.limit:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out, encoding="utf-8") if l.strip()}
    todo = sorted((r for r in rows if r["id"] not in done), key=lambda r: len(r["text"]))  # similar lengths share a batch
    print(f"{len(done)} already labelled, {len(todo)} to do", flush=True)

    from agents import load_agent
    agent = load_agent(args.model)
    t0 = time.time()
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(0, len(todo), args.batch):
            chunk = todo[i:i + args.batch]
            texts = [r["text"][: args.max_chars] for r in chunk]
            try:
                outs = agent.predict_batch(texts, EXTRA)
            except Exception as e:  # noqa: BLE001  (one bad batch: fall back to single requests)
                print("batch failed, one by one:", type(e).__name__, str(e)[:120], flush=True)
                outs = [agent.predict(t, EXTRA)["answers"] for t in texts]
            for r, t, a in zip(chunk, texts, outs):
                ans = {q: (a[q]["noul"] if EXTRA[q]["type"] == "noul" else [a[q]["probabilities"][str(k)] for k in range(len(EXTRA[q]["criteria"]))])
                       for q in EXTRA}
                f.write(json.dumps({"id": r["id"], "source": r["source"], "business": r["business"], "text": t, "answers": ans}, ensure_ascii=False) + "\n")
            f.flush()
            n = i + len(chunk)
            if n % 400 < args.batch:
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el / 60:.1f} min  {el / n:.2f} s/text  eta {el / n * (len(todo) - n) / 60:.0f} min", flush=True)
    print(f"labelled {len(todo)} in {(time.time() - t0) / 60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
