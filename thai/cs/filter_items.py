"""Keep the items of some sources from an items .pt (for a short second pass on real rows only).

    python filter_items.py --inp /work/thai/data/cs/cs9_items.pt --out /work/thai/data/cs/pantip9_items.pt --prefix pantip
"""
import argparse
from collections import Counter

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--inp", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--prefix", default="pantip", help="keep items whose source starts with this")
args = ap.parse_args()
items = torch.load(args.inp)
keep = [i for i in items if str(i.get("source", "")).startswith(args.prefix)]
torch.save(keep, args.out)
print(f"{len(keep)} / {len(items)} items kept:", dict(Counter(i["source"] for i in keep)), "by type", dict(Counter(i["qtype"] for i in keep)))
