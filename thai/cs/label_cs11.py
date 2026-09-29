"""Run 11 items: the real Pantip rows again (as run 10), plus
  - short chat versions of them (shorten_pantip.py, set "train"): human intent / department / urgency, no teacher;
  - human frustration labels (data_domain/frustration_train.json, 300 rows) instead of the teacher's, and the teacher's
    frustration dropped on the other Pantip rows (run 8 over-scores frustration on real text);
  - hard examples: in-scope rows (long or short) that --hard-model calls `other` get their intent question --hard-repeat
    more times;
  - replay: a sample of the synthetic cs9 items, so the synthetic-trained abilities are not forgotten.

    python label_cs11.py --hard-model /work/thai/out/laya-th-run8
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402
from cs_questions import ORDER, question_set  # noqa: E402
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402


def one_hot(q, label, smooth):
    keys = list(range(len(q["criteria"]))) if q["type"] == "score" else list(q["criteria"])
    v = [smooth / len(keys)] * len(keys)
    v[keys.index(label)] += 1.0 - smooth
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", default="/work/thai/data/cs")
    ap.add_argument("--short", default="/work/thai/data/cs/pantip_short.jsonl")
    ap.add_argument("--frustration", default="/work/thai/data_domain/frustration_train.json")
    ap.add_argument("--out", default="/work/thai/data/cs/cs11_items.pt")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run8")
    ap.add_argument("--hard-model", default="/work/thai/out/laya-th-run8")
    ap.add_argument("--long-repeat", type=int, default=4, help="labelled questions of a real post, times")
    ap.add_argument("--short-repeat", type=int, default=2, help="labelled questions of a short version, times")
    ap.add_argument("--hard-repeat", type=int, default=3, help="extra intent items for in-scope rows the hard model calls other")
    ap.add_argument("--replay", type=int, default=20000, help="synthetic cs9 items")
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    frus = json.load(open(args.frustration, encoding="utf-8"))["labels"]

    # ---- records: long Pantip rows from cs9 (human labels + teacher's shared questions), short versions (human labels only)
    recs = []
    for line in open(Path(args.cs) / "cs9.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if not r["source"].startswith("pantip"):
            continue
        biz = r["id"].split("-")[1]
        r["questions"] = question_set(biz)
        r["kind"] = "long"
        r["targets"].pop("frustration", None)  # the teacher's frustration: replaced by human labels where we have them
        if r["id"] in frus:
            r["labels"]["frustration"] = frus[r["id"]]
            r["targets"]["frustration"] = one_hot(r["questions"]["frustration"], frus[r["id"]], args.smooth)
        recs.append(r)
    n_long = len(recs)
    for line in open(args.short, encoding="utf-8"):
        s = json.loads(line)
        if s["set"] != "train":
            continue
        labels = dict(s["labels"])
        if labels["intent"] != "other":
            labels["business"] = s["business"]
        q = question_set(s["business"])
        recs.append({"id": f'{s["id"]}-s{s["variant"]}', "source": f"short_{s['business']}", "state": s["text"], "labels": labels,
                     "questions": q, "kind": "short", "targets": {k: one_hot(q[k], v, args.smooth) for k, v in labels.items()}})
    print(f"{n_long} long Pantip records, {len(recs) - n_long} short versions; {sum('frustration' in r['labels'] for r in recs)} with human frustration", flush=True)

    # ---- hard examples: in-scope rows the current model calls `other`
    agent = laya.Agent(args.hard_model, device="cuda")
    hard = Counter()
    for r in recs:
        if r["labels"]["intent"] == "other":
            continue
        pred = agent.predict(r["state"], {"intent": r["questions"]["intent"]})["answers"]["intent"]["choice"]
        r["hard"] = pred == "other"
        hard[(r["kind"], r["hard"])] += 1
    del agent
    print(f"in-scope rows the hard model calls other: long {hard[('long', True)]}/{hard[('long', True)] + hard[('long', False)]}, "
          f"short {hard[('short', True)]}/{hard[('short', True)] + hard[('short', False)]}", flush=True)

    # ---- items
    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    items, dropped, by = [], 0, Counter()
    for r in recs:
        rep = args.long_repeat if r["kind"] == "long" else args.short_repeat
        qids = [q for q in r["labels"] for _ in range(rep)] + [q for q in ORDER if q not in r["labels"] and q in r["targets"]]
        if r.get("hard"):
            qids += ["intent"] * args.hard_repeat
        for qid in qids:
            q = r["questions"][qid]
            qi = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria") or ({} if q["type"] == "noul" else None)}
            seq, markers = build_sequence(tok, r["state"], qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                dropped += 1
                continue
            t = r["targets"][qid]
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]], "target": t,
                          "label": max(range(len(t)), key=t.__getitem__), "source": f"{r['kind']}_{qid}"})
            by[f"{r['kind']}_{qid}"] += 1
    n_real = len(items)
    pool = [i for i in torch.load(Path(args.cs) / "cs9_items.pt", weights_only=True) if not str(i["source"]).startswith("pantip")]
    items += rng.sample(pool, min(args.replay, len(pool)))
    del pool
    rng.shuffle(items)
    torch.save(items, args.out)
    print(json.dumps({"items": len(items), "real_items": n_real, "replay": len(items) - n_real, "dropped": dropped,
                      "by": dict(sorted(by.items()))}, indent=1), flush=True)


if __name__ == "__main__":
    main()
