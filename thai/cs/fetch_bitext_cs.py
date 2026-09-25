"""Fetch the three Bitext customer-service sets (telco, retail banking, insurance; cdla-sharing-1.0) and sample
--per-intent utterances of each intent into one JSONL: {text_en, business, category, intent, flags}.

    python fetch_bitext_cs.py --out /work/thai/data/cs --per-intent 300
"""
import argparse
import collections
import json
import random
from pathlib import Path

from datasets import load_dataset

SETS = {
    "telecom": "bitext/Bitext-telco-llm-chatbot-training-dataset",
    "banking": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
    "insurance": "bitext/Bitext-insurance-llm-chatbot-training-dataset",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/thai/data/cs")
    ap.add_argument("--per-intent", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, manifest = [], {}
    for business, repo in SETS.items():
        ds = load_dataset(repo, split="train")
        by_intent = collections.defaultdict(list)
        for r in ds:
            text = (r.get("instruction") or r.get("text") or "").strip()
            if len(text) < 8:
                continue
            by_intent[(r["category"].strip(), r["intent"].strip())].append({"text_en": text, "flags": r.get("flags", "")})
        n = 0
        for (cat, intent), items in sorted(by_intent.items()):
            rng.shuffle(items)
            for it in items[: args.per_intent]:
                rows.append({**it, "business": business, "category": cat, "intent": intent})
                n += 1
        manifest[business] = {"rows": len(ds), "intents": len(by_intent), "sampled": n}
        print(business, manifest[business], flush=True)
    rng.shuffle(rows)
    with open(out / "bitext_cs_en.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "bitext_cs_manifest.json").write_text(json.dumps({"per_intent": args.per_intent, "total": len(rows), **manifest}, indent=2))
    print("total", len(rows))


if __name__ == "__main__":
    main()
