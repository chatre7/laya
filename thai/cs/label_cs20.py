"""Run 20 items: all four new questions (cs_questions.EXTRA: churn_threat, external_threat, third_party, contact_effort),
labelled by Underdog-Saluki-27B (label_new_gen.py --cue-only, saluki_label.sh) over the 4,080 pool texts that contain a cue
word of any question, plus a replay of the run 16 items. Saluki beat Qwen3-8B on every question against the 307 hand labels
(churn 41/42 right vs 39/51, third_party 19/30 vs 10/17, effort exact 288 vs 266), so its labels replace the run 19 ones.

Per question: every text Saluki calls yes (repeated when there are few), every Saluki "no" that contains a cue word of that
question (capped at --neg-ratio times the yes count, so a rare question is not drowned), then other "no" texts up to that
ratio: first Saluki "no" texts without the cue, then Qwen3-8B "no" texts outside the cue pool (texts with no cue word are
almost never a yes, so the weaker labeller is safe there). The 307 hand-checked texts stay out: they are the test.

    python label_cs20.py
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
    ap.add_argument("--labels", default="/work/thai/data/cs/new_saluki.jsonl", help="Saluki labels over the cue-word texts")
    ap.add_argument("--labels-rest", default="/work/thai/data/cs/new_gen8.jsonl", help="Qwen3-8B labels over the whole pool; 'no' texts outside the cue pool")
    ap.add_argument("--hand", default="/work/thai/data_domain/new_labels.json")
    ap.add_argument("--out", default="/work/thai/data/cs/cs20_items.pt")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run19")
    ap.add_argument("--questions", default="churn_threat,external_threat,third_party,contact_effort")
    ap.add_argument("--neg-ratio", type=float, default=3.0)
    ap.add_argument("--min-yes", type=int, default=600, help="repeat the yes texts of a question up to about this many items")
    ap.add_argument("--smooth", type=float, default=0.1, help="Saluki is right about 9 times in 10 on the hand check")
    ap.add_argument("--replay", default="/work/thai/data/cs/cs16_items.pt")
    ap.add_argument("--replay-n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    hand = set(json.load(open(args.hand, encoding="utf-8"))["labels"])
    rows = [json.loads(l) for l in open(args.labels, encoding="utf-8")]
    rows = [r for r in rows if r["id"] not in hand]
    cue_ids = {r["id"] for r in rows}
    rest = [json.loads(l) for l in open(args.labels_rest, encoding="utf-8")]
    rest = [r for r in rest if r["id"] not in hand and r["id"] not in cue_ids]
    rng.shuffle(rows)
    rng.shuffle(rest)
    print(f"{len(rows)} Saluki-labelled cue texts outside the hand check {dict(Counter(r['source'] for r in rows))}; "
          f"{len(rest)} other texts with Qwen3-8B labels", flush=True)

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
        n_neg = int(args.neg_ratio * len(yes))
        hard = [r for r in rows if level(r) == 0 and cue.search(r["text"])][:n_neg]
        easy = [r for r in rows if level(r) == 0 and not cue.search(r["text"])] + [r for r in rest if level(r) == 0 and not cue.search(r["text"])]
        n_easy = max(0, n_neg - len(hard))
        rep = min(6, max(1, round(args.min_yes / max(1, len(yes)))))
        picked = [(r, rep) for r in yes] + [(r, rep) for r in hard] + [(r, rep) for r in easy[:n_easy]]
        print(f"{q}: yes {len(yes)} {dict(sorted(Counter(level(r) for r in yes).items()))}, no with a cue word {len(hard)}, other no {min(n_easy, len(easy))}, "
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
