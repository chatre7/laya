"""Print a random sample of each kind from gen_sarcasm.py's output, for a read-through before training."""
import json
import random
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "thai/data/cs/sarcasm_gen.jsonl"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
rows = [json.loads(l) for l in open(path, encoding="utf-8")]
random.seed(1)
for kind in ("sarcastic", "sincere", "plain_negative", "indirect_praise"):
    s = [r for r in rows if r["kind"] == kind]
    for r in random.sample(s, min(n, len(s))):
        print(kind[:5], "|", r["business"][:4], "|", r["text"][:150])
