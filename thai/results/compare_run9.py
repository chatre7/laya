import json
import os

here = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(here, f"{n}.json"), encoding="utf-8"))  # noqa: E731

print("== real Pantip 360 (run3 / run7 / run8 / run9)")
rs = [R(f"{r}_realcs8") for r in ("run3", "run7", "run8", "run9")]
for s in sorted(rs[3]["sources"]):
    print(f"{s:26s} " + " ".join(f"{d['sources'][s]['acc']:7.3f}" for d in rs) + f"  n={rs[3]['sources'][s]['n']}")
for k in ("accuracy", "brier", "ece"):
    print(f"overall {k:12s} " + " ".join(f"{d['overall'][k]:7.3f}" for d in rs))

print("== cs9 held-out (run8 / run9)")
a, b = R("run8_cs9"), R("run9_cs9")
for s in sorted(b["sources"]):
    print(f"{s:26s} {a['sources'].get(s, {}).get('acc', float('nan')):7.3f} {b['sources'][s]['acc']:7.3f}  n={b['sources'][s]['n']}")
for k in ("accuracy", "brier", "ece"):
    print(f"overall {k:12s} {a['overall'][k]:7.3f} {b['overall'][k]:7.3f}")

print("== public (run8 / run9)")
a, b = R("run8"), R("run9")
for s in sorted(b["sources"]):
    print(f"{s:26s} {a['sources'][s]['acc']:7.3f} {b['sources'][s]['acc']:7.3f}")
for k in ("accuracy", "ece"):
    print(f"overall {k:12s} {a['overall'][k]:7.3f} {b['overall'][k]:7.3f}")
print("== krathu-500: run8", R("run8_krathu")["overall"]["accuracy"], "run9", R("run9_krathu")["overall"]["accuracy"])
m = json.load(open(os.path.join(here, "cs9_manifest.json"), encoding="utf-8"))
print("items", m["items"], "pantip items", {k: v for k, v in m["items_by_source"].items() if k.startswith("pantip")}, "minutes(label)", m["minutes"])
