"""Does the model's reading of mood agree with what customers themselves said? Two free checks, no training:

  1. Google Play reviews (fetch_play_reviews.py): the star rating the reviewer gave against the model's `sentiment` (positive /
     neutral / negative / question) and `frustration` (0-2) answers on the review text. Stars 1-2 = unhappy, 4-5 = happy.
  2. Tippawan/thai-ambiguous-sentiment (300 sarcastic / ambiguous Thai sentences with pos / neg labels): `sentiment` again.

    python eval_stars.py --models /work/thai/out/laya-th-run14 --n 3000
"""
import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402
from cs_questions import SHARED  # noqa: E402

QS = {"sentiment": SHARED["sentiment"], "frustration": SHARED["frustration"]}


def auc(pos, neg):
    """P(score of a random unhappy review > score of a random happy one); ties count half."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--reviews", default="/work/thai/data_domain/play_reviews.jsonl")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default="/work/thai/out/stars.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.reviews, encoding="utf-8")]
    rows = [r for r in rows if r.get("score") in (1, 2, 3, 4, 5)]
    rng.shuffle(rows)
    rows = rows[: args.n]
    print(f"{len(rows)} reviews, stars {dict(sorted(Counter(r['score'] for r in rows).items()))}")

    amb = []
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi, HfFileSystem
        repo = "Tippawan/thai-ambiguous-sentiment"
        for f in [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]:
            amb += pq.read_table(f"datasets/{repo}/{f}", filesystem=HfFileSystem()).to_pylist()
        print(f"{len(amb)} ambiguous sentences, labels {dict(Counter(r['label'] for r in amb))}, types {dict(Counter(r['type'] for r in amb))}")
    except Exception as e:  # noqa: BLE001
        print("ambiguous set not loaded:", type(e).__name__, e)

    report = {}
    for m in args.models.split(","):
        agent = laya.Agent(m, device="cuda")
        name = os.path.basename(m)
        table = defaultdict(Counter)        # star -> predicted sentiment
        fr = defaultdict(list)              # star -> frustration scores
        by_biz = defaultdict(lambda: [0, 0])
        for r in rows:
            a = agent.predict(r["text"], QS)["answers"]
            table[r["score"]][a["sentiment"]["choice"]] += 1
            fr[r["score"]].append(a["frustration"]["score"])
            if r["score"] != 3:
                want = "negative" if r["score"] <= 2 else "positive"
                by_biz[r["business"]][0] += a["sentiment"]["choice"] == want
                by_biz[r["business"]][1] += 1
        low, high = fr[1] + fr[2], fr[4] + fr[5]
        n_polar = sum(sum(table[s].values()) for s in (1, 2, 4, 5))
        hit = sum(table[s]["negative"] for s in (1, 2)) + sum(table[s]["positive"] for s in (4, 5))
        flipped = sum(table[s]["positive"] for s in (1, 2)) + sum(table[s]["negative"] for s in (4, 5))
        rep = {"sentiment_matches_stars": round(hit / n_polar, 4), "opposite": round(flipped / n_polar, 4),
               "frustration_mean_by_star": {s: round(sum(fr[s]) / max(1, len(fr[s])), 3) for s in sorted(fr)},
               "frustration_auc_low_vs_high_stars": round(auc(low, high), 4),
               "by_business": {b: round(a / max(1, n), 4) for b, (a, n) in by_biz.items()},
               "table": {s: dict(table[s]) for s in sorted(table)}}
        print(f"\n== {name}: reviews")
        print("stars   n   negative  neutral  positive  question   mean frustration")
        for s in sorted(table):
            n = sum(table[s].values())
            print(f"  {s}   {n:5d}   {table[s]['negative'] / n:7.2f}  {table[s]['neutral'] / n:7.2f}  {table[s]['positive'] / n:8.2f}  {table[s]['question'] / n:8.2f}   {rep['frustration_mean_by_star'][s]:.2f}")
        print(f"1-2 stars called negative or 4-5 stars called positive: {rep['sentiment_matches_stars']:.3f}; the opposite: {rep['opposite']:.3f}; by business {rep['by_business']}")
        print(f"frustration separates 1-2 stars from 4-5 stars with AUC {rep['frustration_auc_low_vs_high_stars']:.3f}")
        if amb:
            c, by_type = Counter(), defaultdict(lambda: [0, 0])
            for r in amb:
                ch = agent.predict(r["sentence"], {"sentiment": SHARED["sentiment"]})["answers"]["sentiment"]["choice"]
                gold = {"neg": "negative", "pos": "positive", "neu": "neutral"}.get(r["label"], r["label"])
                c[gold, ch] += 1
                by_type[r["type"]][0] += ch == gold
                by_type[r["type"]][1] += 1
            acc = sum(v for (g, p), v in c.items() if g == p) / len(amb)
            rep["ambiguous"] = {"accuracy": round(acc, 4), "by_type": {t: [a, n] for t, (a, n) in by_type.items()},
                                "confusion": {f"{g}->{p}": v for (g, p), v in c.items()}}
            print(f"ambiguous / sarcastic sentences: {acc:.3f}; by type " + ", ".join(f"{t} {a}/{n}" for t, (a, n) in by_type.items()))
            print("  gold -> answer:", dict(sorted(rep["ambiguous"]["confusion"].items(), key=lambda kv: -kv[1])))
        report[name] = rep
        del agent
    json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
