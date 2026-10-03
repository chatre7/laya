"""Run 15 items: mood on short Thai service text, sarcasm included.
  - composed contrast sets (gen_sarcasm.py): sarcastic / sincere / plain negative / indirect praise, sentiment (+ frustration
    where defined); 10% held out as sarcasm_gen_eval.jsonl;
  - the hand-checked low-star reviews run 14 read as positive (data_domain/sarcasm_mined.json): sarcasm and plain negatives;
  - star ratings as sentiment labels (mine_sarcasm.py -> play_sentiment.jsonl): 1-2 stars = negative, 4-5 = positive, only
    where the text does not contradict the stars (5-star reviews that complain are common);
  - replay of the run 14 items.
Run 16 adds --wisesight: run 15 had only positive / negative new items and stopped answering "neutral", so the Wisesight
training split (neutral and question mostly, texts of the eval sets excluded) goes in under the same `sentiment` question.

    python label_cs15.py
    python label_cs15.py --wisesight /work/thai/data/cc/wisesight_train.jsonl --out /work/thai/data/cs/cs16_items.pt
Run 17 weights the neutral side more (run 16 got "neutral" back only half-way): 6,000 neutral x2, question x3.
    python label_cs15.py --wisesight ... --ws-neutral 6000 --ws-neutral-repeat 2 --ws-question-repeat 3 --out .../cs17_items.pt
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
from cs_questions import SHARED  # noqa: E402
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402


def one_hot(q, label, smooth):
    keys = list(range(len(q["criteria"]))) if q["type"] == "score" else list(q["criteria"])
    v = [smooth / len(keys)] * len(keys)
    v[keys.index(label)] += 1.0 - smooth
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="/work/thai/data/cs/sarcasm_gen.jsonl")
    ap.add_argument("--mined", default="/work/thai/data_domain/sarcasm_mined.json")
    ap.add_argument("--stars", default="/work/thai/data/cs/play_sentiment.jsonl")
    ap.add_argument("--out", default="/work/thai/data/cs/cs15_items.pt")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run14")
    ap.add_argument("--star-cap", type=int, default=3000, help="reviews per polarity")
    ap.add_argument("--mined-repeat", type=int, default=6)
    ap.add_argument("--replay", default="/work/thai/data/cs/cs14_items.pt")
    ap.add_argument("--replay-n", type=int, default=60000)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wisesight", default="", help="wisesight_train.jsonl (text, label 0 pos / 1 neu / 2 neg / 3 q); off when empty")
    ap.add_argument("--ws-neutral", type=int, default=3000)
    ap.add_argument("--ws-polar", type=int, default=1000, help="positive and negative each")
    ap.add_argument("--ws-question-repeat", type=int, default=2)
    ap.add_argument("--ws-neutral-repeat", type=int, default=1, help="run 17: 2")
    ap.add_argument("--exclude", default="/work/thai/data/eval.jsonl,/work/thai/data/cs/cs9_eval_human.jsonl", help="eval sets whose texts stay out")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    recs = []  # (text, sentiment, frustration or None, repeat, source)

    gen = [json.loads(l) for l in open(args.gen, encoding="utf-8")]
    rng.shuffle(gen)
    n_ev = len(gen) // 10
    with open(Path(args.gen).with_name("sarcasm_gen_eval.jsonl"), "w", encoding="utf-8") as f:
        for r in gen[:n_ev]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in gen[n_ev:]:
        recs.append((r["text"], r["sentiment"], r["frustration"], 2 if r["kind"] == "sarcastic" else 1, f"gen_{r['kind']}"))
    print(f"composed: {len(gen) - n_ev} train {dict(Counter(r['kind'] for r in gen[n_ev:]))}, {n_ev} held out", flush=True)

    mined = json.load(open(args.mined, encoding="utf-8"))["rows"]
    used = {m["id"] for m in mined}
    for m in mined:
        if m["kind"] in ("sarcasm", "plain_negative"):
            recs.append((m["text"], "negative", 1 if m["kind"] == "sarcasm" else None, args.mined_repeat, f"mined_{m['kind']}"))

    stars = [json.loads(l) for l in open(args.stars, encoding="utf-8")]
    rng.shuffle(stars)
    n_star = Counter()
    for r in stars:
        if r["id"] in used or r["score"] == 3:
            continue
        lab = "negative" if r["score"] <= 2 else "positive"
        if r["pred"] == ("positive" if lab == "negative" else "negative") or n_star[lab] >= args.star_cap:
            continue  # text and stars contradict (or the cap is reached)
        n_star[lab] += 1
        recs.append((r["text"], lab, None, 1, f"stars_{lab}"))
    print(f"mined {sum(m['kind'] in ('sarcasm', 'plain_negative') for m in mined)} x{args.mined_repeat}; star-labelled {dict(n_star)}", flush=True)

    if args.wisesight:
        held = set()
        for p in args.exclude.split(","):
            for l in open(p, encoding="utf-8"):
                st = json.loads(l)["state"]
                held.update(v.strip() for v in (st.values() if isinstance(st, dict) else [st]) if isinstance(v, str))
        ws = [json.loads(l) for l in open(args.wisesight, encoding="utf-8")]
        rng.shuffle(ws)
        cap = {"positive": args.ws_polar, "neutral": args.ws_neutral, "negative": args.ws_polar, "question": len(ws)}
        n_ws, n_held = Counter(), 0
        for r in ws:
            lab = ("positive", "neutral", "negative", "question")[r["label"]]
            if r["text"].strip() in held:
                n_held += 1
            elif len(r["text"].strip()) >= 8 and n_ws[lab] < cap[lab]:
                n_ws[lab] += 1
                rep = {"question": args.ws_question_repeat, "neutral": args.ws_neutral_repeat}.get(lab, 1)
                recs.append((r["text"].strip(), lab, None, rep, f"wisesight_{lab}"))
        print(f"wisesight {dict(n_ws)} (question x{args.ws_question_repeat}, neutral x{args.ws_neutral_repeat}), {n_held} skipped as eval texts", flush=True)

    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    items, by = [], Counter()
    for text, sent, frus, rep, src in recs:
        for qid, lab in (("sentiment", sent), ("frustration", frus)):
            if lab is None:
                continue
            q = SHARED[qid]
            qi = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
            seq, markers = build_sequence(tok, text, qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                continue
            t = one_hot(q, lab, args.smooth)
            item = {"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]], "target": t, "label": max(range(len(t)), key=t.__getitem__), "source": src}
            n = rep if qid == "sentiment" else 1
            items += [item] * n
            by[f"{src}:{qid}"] += n
    n_new = len(items)
    pool = torch.load(args.replay, weights_only=True)
    items += rng.sample(pool, min(args.replay_n, len(pool)))
    del pool
    rng.shuffle(items)
    torch.save(items, args.out)
    print(json.dumps({"new_items": n_new, "replay": len(items) - n_new, "items": len(items), "by": dict(sorted(by.items()))}, indent=1), flush=True)


if __name__ == "__main__":
    main()
