"""Run 14 items, a balanced mix:
  - forum / social posts without hand labels (label_pantip_llm.py), kept per question only where the LLM and the student agree
    (on the 360 hand-labelled posts each alone is ~57% right on intent, agreed labels 80%; department 88%, urgency 90%);
  - Google Play reviews with the LLM's labels (label_play_llm.py), capped harder than in run 13 so they do not pull the model
    towards one register;
  - replay of run 11 items (hand-labelled posts long and short, human frustration, synthetic).

    python label_cs14.py
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402
from cs_questions import question_set  # noqa: E402
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402


def one_hot(q, label, smooth):
    keys = list(range(len(q["criteria"]))) if q["type"] == "score" else list(q["criteria"])
    v = [smooth / len(keys)] * len(keys)
    v[keys.index(label)] += 1.0 - smooth
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts", default="/work/thai/data/cs/pantip_llm.jsonl")
    ap.add_argument("--reviews", default="/work/thai/data/cs/play_llm.jsonl")
    ap.add_argument("--out", default="/work/thai/data/cs/cs14_items.pt")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run11")
    ap.add_argument("--review-cap", type=int, default=800, help="reviews per (business, intent)")
    ap.add_argument("--post-intent-repeat", type=int, default=3)
    ap.add_argument("--replay", default="/work/thai/data/cs/cs11_items.pt")
    ap.add_argument("--replay-n", type=int, default=60000)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # ---- posts: per-question agreement between the LLM and the student
    posts = [json.loads(l) for l in open(args.posts, encoding="utf-8")]
    agent = laya.Agent(args.student, device="cuda")
    agree = Counter()
    recs = []  # (text, business, {qid: label}, {qid: repeat})
    for r in posts:
        q = question_set(r["business"])
        a = agent.predict(r["text"], {k: q[k] for k in ("intent", "department", "urgency")})["answers"]
        s = {"intent": a["intent"]["choice"], "department": a["department"]["choice"], "urgency": round(a["urgency"]["score"])}
        labels = {k: v for k, v in r["labels"].items() if s[k] == v}
        for k in labels:
            agree[k] += 1
        if "intent" in labels and labels["intent"] != "other":
            labels["business"] = r["business"]
        if labels:
            recs.append((r["text"], r["business"], labels, {"intent": args.post_intent_repeat}))
    del agent
    print(f"{len(posts)} posts; agreed labels {dict(agree)}; {len(recs)} posts keep at least one question", flush=True)

    # ---- reviews, capped
    reviews = [json.loads(l) for l in open(args.reviews, encoding="utf-8")]
    rng.shuffle(reviews)
    seen, n_rev = Counter(), 0
    for r in reviews:
        k = (r["business"], r["labels"]["intent"])
        if seen[k] >= args.review_cap:
            continue
        seen[k] += 1
        n_rev += 1
        labels = dict(r["labels"])
        if labels["intent"] != "other":
            labels["business"] = r["business"]
        recs.append((r["text"], r["business"], labels, {"intent": 2}))
    print(f"{len(reviews)} reviews -> {n_rev} after cap {args.review_cap}", flush=True)

    # ---- items
    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    items, dropped = [], 0
    for text, biz, labels, rep in recs:
        q = question_set(biz)
        for qid, lab in labels.items():
            qq = q[qid]
            qi = {"t": qq["type"], "ins": qq["instructions"], "crit": qq.get("criteria")}
            seq, markers = build_sequence(tok, text, qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                dropped += 1
                continue
            t = one_hot(qq, lab, args.smooth)
            item = {"ids": seq, "markers": markers, "qtype": QTYPES[qq["type"]], "target": t,
                    "label": max(range(len(t)), key=t.__getitem__), "source": "llm"}
            items += [item] * rep.get(qid, 1)
    n_new = len(items)
    pool = torch.load(args.replay, weights_only=True)
    items += rng.sample(pool, min(args.replay_n, len(pool)))
    del pool
    rng.shuffle(items)
    torch.save(items, args.out)
    print(json.dumps({"posts_kept": len(recs) - n_rev, "reviews": n_rev, "new_items": n_new, "replay": len(items) - n_new, "items": len(items),
                      "dropped": dropped}, indent=1), flush=True)


if __name__ == "__main__":
    main()
