"""Run 18 items: the Thai part of the `train` split of hagsmand1/laya-thai-decisions, plus a replay of the run 16 items.

That dataset is everything our own items are not: states are objects with distractor fields, instructions point at a field
in backticks and come in Thai or English, option keys are letters / Thai words / slugs in shuffled order, scales run in both
directions with 3-4 levels, yes/no questions are also asked negated and relabelled. Each question carries a soft target (gold
blended with an LLM teacher), used here as is.

Left out: `thaisum` (scraped news, publisher copyright, per the dataset card), the non-Thai rows, and any row whose text is
in our own eval sets (their rows come from the upstream train splits, and so does part of our held-out set).

    python label_cs18.py
"""
import argparse
import json
import os
import random
import sys
from collections import Counter

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from laya.agent import Agent, _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402

REPO = "hagsmand1/laya-thai-decisions"


def squash(text):
    return "".join(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/thai/data/cs/cs18_items.pt")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run16")
    ap.add_argument("--skip-sources", default="thaisum")
    ap.add_argument("--exclude", default="/work/thai/data/eval.jsonl,/work/thai/data/cs/cs9_eval_human.jsonl", help="eval sets whose texts stay out")
    ap.add_argument("--replay", default="/work/thai/data/cs/cs16_items.pt")
    ap.add_argument("--replay-n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    held = set()
    for p in args.exclude.split(","):
        for l in open(p, encoding="utf-8"):
            st = json.loads(l)["state"]
            held.update(squash(v) for v in (st.values() if isinstance(st, dict) else [st]) if isinstance(v, str))

    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    skip = set(args.skip_sources.split(","))
    items, by, dropped = [], Counter(), Counter()
    for r in load_dataset(REPO, split="train"):
        if r["lang"] != "th" or r["source"] in skip or not r["target"]:
            dropped["not Thai" if r["lang"] != "th" else r["source"] if r["source"] in skip else "no target"] += 1
            continue
        if squash(r["text"]) in held:
            dropped["in our eval sets"] += 1
            continue
        state, questions, target = json.loads(r["state"]), json.loads(r["questions"]), json.loads(r["target"])
        for qid, t in target.items():
            q = questions[qid]
            qi = Agent._to_internal(q)
            seq, markers = build_sequence(tok, state, qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                dropped["options over the head budget"] += 1
                continue
            keys = [str(i) for i in range(len(q["criteria"]))] if q["type"] == "score" else ["false", "true"] if q["type"] == "noul" else list(q["criteria"])
            v = [float(t[k]) for k in keys]
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]], "target": v, "label": max(range(len(v)), key=v.__getitem__),
                          "source": f"ltd_{r['source']}"})
            by[f"{r['source']}:{q['type']}"] += 1
    n_new = len(items)
    pool = torch.load(args.replay, weights_only=True)
    items += rng.sample(pool, min(args.replay_n, len(pool)))
    del pool
    rng.shuffle(items)
    torch.save(items, args.out)
    print(json.dumps({"new_items": n_new, "replay": len(items) - n_new, "items": len(items), "dropped": dict(dropped), "by": dict(sorted(by.items()))}, indent=1), flush=True)


if __name__ == "__main__":
    main()
