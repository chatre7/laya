import json
import os

here = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(here, f"{n}.json"), encoding="utf-8"))  # noqa: E731

rs = {k: R(f"{k}_cs7") for k in ("run5", "run6", "run7")}
print("== cs7 held-out (run5 / run6 / run7)")
for src in rs["run7"]["sources"]:
    print(f"{src:22s}" + " ".join(f"{rs[k]['sources'][src]['acc']:>8.3f}" for k in rs) + f"  n={rs['run7']['sources'][src]['n']}")
for key in ("accuracy", "brier", "ece"):
    print(f"overall {key:14s}" + " ".join(f"{rs[k]['overall'][key]:>8.3f}" for k in rs))
for t in ("choice", "noul", "score"):
    print(f"teacher agree {t:8s}" + " ".join(f"{rs[k]['teacher'][t]['argmax_agreement']:>8.3f}" for k in rs))
p = {k: R(k) for k in ("run6", "run7")}
print("== public (run6 / run7)")
for src in p["run7"]["sources"]:
    print(f"{src:22s}" + " ".join(f"{p[k]['sources'][src]['acc']:>8.3f}" for k in p))
for key in ("accuracy", "ece"):
    print(f"overall {key:14s}" + " ".join(f"{p[k]['overall'][key]:>8.3f}" for k in p))
m = json.load(open(os.path.join(here, "cs7_manifest.json"), encoding="utf-8"))
print("items", m["items"], m["by_source"])
