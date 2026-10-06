"""Run 19 items: three questions laya has never been asked (cs_questions.EXTRA: churn_threat, external_threat,
contact_effort), labelled over our real Thai service text by a generative LLM that reads written rules (label_new_gen.py,
Qwen3-8B), plus a replay of the run 16 items. third_party was dropped: the labeller got 10 of 17 right on the hand check.

A "yes" is rare, so per question: every text the labeller calls yes (repeated when there are few), every "no" that contains
a cue word of that question (the hard ones: cancel a package, a police report about a scammer), and random other "no" up to
--neg-ratio times the yes count. The 307 hand-checked texts (data_domain/new_labels.json) stay out: they are the test.

    python label_cs19.py
"""
import argparse
import json
import os
import random
import re
import sys
from collections import Counter

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from cs_questions import EXTRA  # noqa: E402
from laya.agent import Agent, _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402
from sample_new_check2 import CUES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/work/thai/data/cs/new_gen8.jsonl")
    ap.add_argument("--hand", default="/work/thai/data_domain/new_labels.json")
    ap.add_argument("--out", default="/work/thai/data/cs/cs19_items.pt")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run16")
    ap.add_argument("--questions", default="churn_threat,external_threat,contact_effort")
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--min-yes", type=int, default=600, help="repeat the yes texts of a question up to about this many items")
    ap.add_argument("--smooth", type=float, default=0.15, help="the labeller is right about 3 times in 4 on hard texts")
    ap.add_argument("--replay", default="/work/thai/data/cs/cs16_items.pt")
    ap.add_argument("--replay-n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    hand = set(json.load(open(args.hand, encoding="utf-8"))["labels"])
    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    rows = [r for r in rows if r["id"] not in hand]
    rng.shuffle(rows)
    print(f"{len(rows)} labelled texts outside the hand check {dict(Counter(r['source'] for r in rows))}", flush=True)

    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    items, by = [], Counter()
    for q in args.questions.split(","):
        d = EXTRA[q]
        qi = Agent._to_internal(d)
        k = len(render_options(qi))
        level = (lambda r: int(r["answers"][q] > 0.5)) if d["type"] == "noul" else (lambda r: max(range(k), key=r["answers"][q].__getitem__))
        cue = re.compile(CUES[q], re.I)
        yes = [r for r in rows if level(r) > 0]
        hard = [r for r in rows if level(r) == 0 and cue.search(r["text"])]
        easy = [r for r in rows if level(r) == 0 and not cue.search(r["text"])]
        n_no = max(0, int(args.neg_ratio * len(yes)) - len(hard))
        rep = min(6, max(1, round(args.min_yes / max(1, len(yes)))))
        picked = [(r, rep) for r in yes] + [(r, rep) for r in hard] + [(r, rep) for r in easy[:n_no]]
        print(f"{q}: yes {len(yes)} {dict(sorted(Counter(level(r) for r in yes).items()))}, no with a cue word {len(hard)}, other no {min(n_no, len(easy))}, "
              f"each x{rep}", flush=True)
        for r, n in picked:
            seq, markers = build_sequence(tok, r["text"], qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != k:
                continue
            lab = level(r)
            t = [args.smooth / k] * k
            t[lab] += 1.0 - args.smooth
            items += [{"ids": seq, "markers": markers, "qtype": QTYPES[d["type"]], "target": t, "label": lab, "source": f"new_{q}"}] * n
            by[f"{q}:{lab}"] += n
    n_new = len(items)
    pool = torch.load(args.replay, weights_only=True)
    items += rng.sample(pool, min(args.replay_n, len(pool)))
    del pool
    rng.shuffle(items)
    torch.save(items, args.out)
    print(json.dumps({"new_items": n_new, "replay": len(items) - n_new, "items": len(items), "by": dict(sorted(by.items()))}, indent=1), flush=True)


if __name__ == "__main__":
    main()
