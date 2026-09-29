import json
import os

here = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(here, f"{n}.json"), encoding="utf-8"))  # noqa: E731
ex = lambda n: os.path.exists(os.path.join(here, f"{n}.json"))  # noqa: E731

runs = ("run8", "run9", "run10")
print("== real Pantip 360 (" + " / ".join(runs) + ")")
rs = [R(f"{r}_realcs8") for r in runs]
for s in sorted(rs[0]["sources"]):
    print(f"{s:22s} " + " ".join(f"{d['sources'][s]['acc']:7.3f}" for d in rs))
for k in ("accuracy", "brier", "ece"):
    print(f"overall {k:12s} " + " ".join(f"{d['overall'][k]:7.3f}" for d in rs))
if ex("run10_cs9"):
    print("== cs9 held-out (run8 / run9 / run10)")
    cs = [R(f"{r}_cs9") for r in runs]
    for s in sorted(cs[2]["sources"]):
        print(f"{s:22s} " + " ".join(f"{d['sources'].get(s, {}).get('acc', float('nan')):7.3f}" for d in cs))
    for k in ("accuracy", "ece"):
        print(f"overall {k:12s} " + " ".join(f"{d['overall'][k]:7.3f}" for d in cs))
if ex("run10"):
    print("== public (run8 / run9 / run10)")
    ps = [R(r) for r in runs]
    for s in sorted(ps[2]["sources"]):
        print(f"{s:22s} " + " ".join(f"{d['sources'][s]['acc']:7.3f}" for d in ps))
    for k in ("accuracy", "ece"):
        print(f"overall {k:12s} " + " ".join(f"{d['overall'][k]:7.3f}" for d in ps))
