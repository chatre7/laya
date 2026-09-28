import json
import os

here = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(here, f"{n}.json"), encoding="utf-8"))  # noqa: E731

print("== cs8 held-out (run7 / run8)")
a, b = R("run7_cs8"), R("run8_cs8")
for s in sorted(b["sources"]):
    print(f"{s:26s} {a['sources'].get(s, {}).get('acc', float('nan')):7.3f} {b['sources'][s]['acc']:7.3f}  n={b['sources'][s]['n']}")
for k in ("accuracy", "brier", "ece"):
    print(f"overall {k:12s} {a['overall'][k]:7.3f} {b['overall'][k]:7.3f}")
for t in ("choice", "noul", "score"):
    print(f"teacher agree {t:8s} {a['teacher'][t]['argmax_agreement']:7.3f} {b['teacher'][t]['argmax_agreement']:7.3f}")

print("== real Pantip 360, run 8 intent lists (run3 / run7 / run8)")
rs = [R(f"{r}_realcs8") for r in ("run3", "run7", "run8")]
for s in sorted(rs[2]["sources"]):
    print(f"{s:26s} " + " ".join(f"{d['sources'][s]['acc']:7.3f}" for d in rs) + f"  n={rs[2]['sources'][s]['n']}")
for k in ("accuracy", "ece"):
    print(f"overall {k:12s} " + " ".join(f"{d['overall'][k]:7.3f}" for d in rs))
print("   per-source keys:", sorted(list(rs[2]["sources"].values())[0].keys()))

print("== krathu-500 sentiment (run3/5/6/7/8); teacher 0.660")
for r in ("run3", "run5", "run6", "run7", "run8"):
    d = R(f"{r}_krathu")
    print(f"{r:6s} acc {d['overall']['accuracy']:.3f}  ece {d['overall']['ece']:.3f}")

print("== public (run7 / run8)")
a, b = R("run7"), R("run8")
for s in sorted(b["sources"]):
    print(f"{s:26s} {a['sources'][s]['acc']:7.3f} {b['sources'][s]['acc']:7.3f}")
for k in ("accuracy", "ece"):
    print(f"overall {k:12s} {a['overall'][k]:7.3f} {b['overall'][k]:7.3f}")
m = json.load(open(os.path.join(here, "cs8_manifest.json"), encoding="utf-8"))
print("items", m["items"], "gated_out", m["gated_out"], "minutes(label)", m["minutes"])
