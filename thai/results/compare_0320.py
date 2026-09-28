"""laya 0.3.5 (run8.sh) vs laya 0.3.20 (check_0320.sh) on the same run 8 checkpoint: numbers must match."""
import json
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(here, f"{n}.json"), encoding="utf-8"))  # noqa: E731
for name, old, new in (("real Pantip 360", "run8_realcs8", "run8_realcs8_0320"), ("public 2,455", "run8", "run8_0320")):
    if not os.path.exists(os.path.join(here, f"{new}.json")):
        print(f"== {name}: {new}.json not there yet")
        continue
    a, b = R(old), R(new)
    print(f"== {name}: 0.3.5 vs 0.3.20")
    worst = 0.0
    for s in sorted(a["sources"]):
        x, y = a["sources"][s]["acc"], b["sources"][s]["acc"]
        worst = max(worst, abs(x - y))
        print(f"{s:22s} {x:.4f} {y:.4f}  brier {a['sources'][s]['brier']:.4f} {b['sources'][s]['brier']:.4f}")
    for k in ("accuracy", "brier", "ece"):
        print(f"overall {k:9s} {a['overall'][k]:.4f} {b['overall'][k]:.4f}")
    print(f"largest per-source accuracy difference: {worst:.4f}")
sys.exit(0)
