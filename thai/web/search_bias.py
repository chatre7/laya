"""Why does web1 pick the Search button over the search box? Count, in the Mind2Web train records, the steps whose options hold
both a search-like textbox and a search-like button, and which of the two (or neither) is the target, split by whether the
previous action already typed into a textbox.

    python3 thai/web/search_bias.py
"""
import json
import re
from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "web"
role = re.compile(r'^\[\d+\] \S+ \((\w+)\) "([^"]*)"')
c = Counter()
for line in open(DATA / "m2w_train.jsonl", encoding="utf-8"):
    r = json.loads(line)
    parsed = [(o, *role.match(o).groups()) for o in r["options"] if role.match(o)]
    box = [o for o, ro, n in parsed if ro == "textbox" and "search" in n.lower()]
    btn = [o for o, ro, n in parsed if ro == "button" and "search" in n.lower()]
    if not (box and btn):
        continue
    typed = bool(r["previous"]) and "-> TYPE" in r["previous"][-1]
    who = "box" if r["target"] in box else "button" if r["target"] in btn else "other"
    c[("after TYPE" if typed else "no TYPE just before", who)] += 1
for k, v in sorted(c.items()):
    print(f"{k[0]:22s} target={k[1]:7s} {v}")
