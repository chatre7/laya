"""Print Mind2Web eval accuracy per model / split / language from thai/out/{m}_m2w_{split}_{lang}.json.

    python3 thai/web/summary_web1.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "out"
print(f"{'split':14s} {'lang':4s} {'n':>5s} {'run3':>6s} {'web1':>6s}")
for split in ("test_task", "test_website", "test_domain"):
    for lang in ("th", "en"):
        r = {m: json.load(open(OUT / f"{m}_m2w_{split}_{lang}.json"))["overall"] for m in ("run3", "web1")}
        print(f"{split:14s} {lang:4s} {r['web1']['decisions']:5d} {r['run3']['accuracy']:6.3f} {r['web1']['accuracy']:6.3f}")
