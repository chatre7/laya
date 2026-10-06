"""Pick the texts for the hand check of the new-question teacher (label_new_llm.py pilot). The new questions are rare
events, so a random sample would hold almost no "yes": per question take texts the teacher is sure are yes, texts it is
unsure about and texts it is sure are no (for contact_effort: per most probable level), then label the union for all four
questions. Writes the sample without the teacher's answers and a print file to read.

    python sample_new_check.py        # -> data_domain/new_check_sample.jsonl, data_domain/_print_new_check.txt
"""
import argparse
import json
import random
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs_questions import EXTRA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="/work/thai/data/cs/new_pilot.jsonl")
    ap.add_argument("--out", default="/work/thai/data_domain/new_check_sample.jsonl")
    ap.add_argument("--print", dest="print_file", default="/work/thai/data_domain/_print_new_check.txt")
    ap.add_argument("--yes", type=int, default=30)
    ap.add_argument("--unsure", type=int, default=15)
    ap.add_argument("--no", type=int, default=20)
    ap.add_argument("--per-level", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.pilot, encoding="utf-8")]
    rng.shuffle(rows)
    picked, why = {}, Counter()

    def take(cands, n, tag):
        got = 0
        for r in cands:
            if got >= n:
                break
            if r["id"] not in picked:
                picked[r["id"]] = r
            why[tag] += 1
            got += 1

    for q, d in EXTRA.items():
        if d["type"] == "noul":
            p = lambda r, q=q: r["answers"][q]  # noqa: E731
            print(q, "teacher yes>0.5:", sum(p(r) > 0.5 for r in rows), " 0.2-0.5:", sum(0.2 <= p(r) <= 0.5 for r in rows), " of", len(rows))
            take([r for r in rows if p(r) > 0.5], args.yes, f"{q}:yes")
            take([r for r in rows if 0.2 <= p(r) <= 0.5], args.unsure, f"{q}:unsure")
            take([r for r in rows if p(r) < 0.2], args.no, f"{q}:no")
        else:
            top = lambda r, q=q: max(range(len(r["answers"][q])), key=r["answers"][q].__getitem__)  # noqa: E731
            print(q, "teacher levels:", dict(sorted(Counter(top(r) for r in rows).items())))
            for lv in range(len(d["criteria"])):
                take([r for r in rows if top(r) == lv], args.per_level, f"{q}:{lv}")
    sample = list(picked.values())
    rng.shuffle(sample)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps({"id": r["id"], "source": r["source"], "business": r["business"], "text": r["text"]}, ensure_ascii=False) + "\n")
    with open(args.print_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(sample):
            f.write(f"{i}|{r['id']}|{r['text'][:420]}\n")
    print(f"{len(sample)} texts to label {dict(why)} -> {args.out}")


if __name__ == "__main__":
    main()
