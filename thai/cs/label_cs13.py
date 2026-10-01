"""Run 13 items: Google Play reviews labelled by the LLM (label_play_llm.py: intent, department, urgency; 74% / 72% / 78%
against the 300 hand labels) + replay of run 11 items. Frequent intents (praise = other, app problems) are capped per business
so they do not swamp the rest.

    python label_cs13.py
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
    ap.add_argument("--inp", default="/work/thai/data/cs/play_llm.jsonl")
    ap.add_argument("--out", default="/work/thai/data/cs/cs13_items.pt")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run11")
    ap.add_argument("--cap", type=int, default=1500, help="rows per (business, intent)")
    ap.add_argument("--replay", default="/work/thai/data/cs/cs11_items.pt")
    ap.add_argument("--replay-n", type=int, default=50000)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    rng.shuffle(rows)
    seen, kept = Counter(), []
    for r in rows:
        k = (r["business"], r["labels"]["intent"])
        if seen[k] < args.cap:
            seen[k] += 1
            kept.append(r)
    print(f"{len(rows)} labelled reviews -> {len(kept)} after cap {args.cap}; top {seen.most_common(8)}", flush=True)
    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    items, dropped = [], 0
    for r in kept:
        q = question_set(r["business"])
        labels = dict(r["labels"])
        if labels["intent"] != "other":
            labels["business"] = r["business"]
        for qid in ["intent", "intent"] + [k for k in labels if k != "intent"]:
            qq = q[qid]
            qi = {"t": qq["type"], "ins": qq["instructions"], "crit": qq.get("criteria")}
            seq, markers = build_sequence(tok, r["text"], qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                dropped += 1
                continue
            t = one_hot(qq, labels[qid], args.smooth)
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[qq["type"]], "target": t,
                          "label": max(range(len(t)), key=t.__getitem__), "source": f"playllm_{r['business']}"})
    n_play = len(items)
    pool = torch.load(args.replay, weights_only=True)
    items += rng.sample(pool, min(args.replay_n, len(pool)))
    del pool
    rng.shuffle(items)
    torch.save(items, args.out)
    print(json.dumps({"reviews": len(kept), "play_items": n_play, "replay": len(items) - n_play, "items": len(items), "dropped": dropped}, indent=1), flush=True)


if __name__ == "__main__":
    main()
