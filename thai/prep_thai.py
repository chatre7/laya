"""Build Thai decision data for fine-tuning laya-multilingual.

1. Runs the upstream OpenThai-SystemOne converters (scripts/03_decision_data.py) for the Thai sources,
   which turn public Thai datasets into unified records {state, questions, labels}.
2. Tokenises every (state, question) pair with laya's build_sequence into training items
   (same format as laya's fine-tune notebook) and writes eval records as jsonl.

    python prep_thai.py --out /work/data --limit 1500
"""
import argparse
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence, render_options

TRAIN_SOURCES = ["wongnai", "prachathai", "xnli_th", "massive_th", "thai_toxicity"]
EVAL_ONLY = ["wisesight", "sib200_th"]  # upstream never trains on these


def load_converters():
    spec = importlib.util.spec_from_file_location("decision_data", "/work/thai/ots/scripts/03_decision_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def record_targets(q, label):
    """Hard label -> target distribution in laya option order."""
    t, crit = q["type"], q.get("criteria")
    if t == "choice":
        keys = list(crit.keys())
        if label not in keys:
            return None
        return [1.0 if k == label else 0.0 for k in keys]
    if t == "score":
        k = len(crit)
        label = int(label)
        if not 0 <= label < k:
            return None
        return [1.0 if i == label else 0.0 for i in range(k)]
    return [0.0, 1.0] if bool(label) else [1.0, 0.0]


def to_item(tok, cfg, state, q, label):
    target = record_targets(q, label)
    if target is None:
        return None
    crit = q.get("criteria") if q["type"] != "noul" else (q.get("criteria") or {})
    qi = {"t": q["type"], "ins": q["instructions"], "crit": crit}
    seq, markers = build_sequence(tok, state, qi, cfg["max_len"], cfg["head_max_len"])
    if len(markers) != len(render_options(qi)):
        return None
    return {"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]], "target": target, "label": target.index(1.0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/thai/data")
    ap.add_argument("--limit", type=int, default=1500, help="records per split per source")
    ap.add_argument("--eval-cap", type=int, default=300, help="eval records per source")
    ap.add_argument("--model", default="convaiinnovations/laya-multilingual")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dd = load_converters()
    recs = {"train": [], "eval": []}
    manifest = {}
    for name in TRAIN_SOURCES + EVAL_ONLY:
        rng = random.Random(f"{args.seed}-{name}")
        print(f"== {name}", flush=True)
        try:
            rs = list(dd.REGISTRY[name](args.limit, rng))
        except Exception as e:  # a dataset that fails to download must not kill the run
            print(f"   FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            manifest[name] = {"error": str(e)[:200]}
            continue
        counts = {}
        for split in ("train", "eval"):
            sub = [r for r in rs if r.split == split]
            if name in EVAL_ONLY and split == "train":
                sub = []
            if split == "eval":
                rng.shuffle(sub)
                sub = sub[: args.eval_cap]
            recs[split].extend(sub)
            counts[split] = len(sub)
        manifest[name] = counts
        print(f"   {counts}", flush=True)

    with open(out / "eval.jsonl", "w", encoding="utf-8") as f:
        for r in recs["eval"]:
            f.write(r.to_json() + "\n")

    model_dir = snapshot_download(args.model, allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"])
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))

    items, dropped, by_type = [], 0, {0: 0, 1: 0, 2: 0}
    for r in recs["train"]:
        for qid, q in r.questions.items():
            if qid not in r.labels:
                continue
            it = to_item(tok, cfg, r.state, q, r.labels[qid])
            if it is None:
                dropped += 1
                continue
            it["source"] = r.source
            items.append(it)
            by_type[it["qtype"]] += 1
    random.Random(args.seed).shuffle(items)
    torch.save(items, out / "train_items.pt")
    manifest["_items"] = {"train_items": len(items), "dropped": dropped,
                          "by_type": {"choice": by_type[0], "score": by_type[1], "noul": by_type[2]},
                          "eval_records": len(recs["eval"]),
                          "mean_len": sum(len(i["ids"]) for i in items) / max(1, len(items))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
