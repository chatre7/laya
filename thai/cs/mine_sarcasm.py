"""Real sarcasm candidates: Google Play reviews with 1-2 stars whose text the model reads as positive. Written out for a
hand check (many are sarcasm; some are a wrong star or a mixed review). Also dumps the model's sentiment for every review,
so run 15 can use star ratings as sentiment labels where text and stars do not contradict.

    python mine_sarcasm.py --model /work/thai/out/laya-th-run14
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402
from cs_questions import SHARED  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/work/thai/out/laya-th-run14")
    ap.add_argument("--reviews", default="/work/thai/data_domain/play_reviews.jsonl")
    ap.add_argument("--out", default="/work/thai/data/cs/play_sentiment.jsonl")
    ap.add_argument("--cand", default="/work/thai/data/cs/sarcasm_candidates.jsonl")
    args = ap.parse_args()
    agent = laya.Agent(args.model, device="cuda")
    rows = [json.loads(l) for l in open(args.reviews, encoding="utf-8")]
    n = 0
    with open(args.out, "w", encoding="utf-8") as f, open(args.cand, "w", encoding="utf-8") as fc:
        for i, r in enumerate(rows):
            a = agent.predict(r["text"], {"sentiment": SHARED["sentiment"]})["answers"]["sentiment"]
            rec = {"id": r["id"], "business": r["business"], "score": r["score"], "text": r["text"], "pred": a["choice"],
                   "p_pos": round(a["probabilities"]["positive"], 3), "p_neg": round(a["probabilities"]["negative"], 3)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if r["score"] in (1, 2) and a["choice"] == "positive":
                fc.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(rows)}", flush=True)
    print(f"{len(rows)} reviews; {n} candidates (1-2 stars, read as positive) -> {args.cand}")


if __name__ == "__main__":
    main()
