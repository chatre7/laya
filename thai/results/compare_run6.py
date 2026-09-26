import json
import os

here = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(here, f"{n}.json"), encoding="utf-8"))  # noqa: E731

rs = {k: R(f"{k}_cs6") for k in ("run3", "run5", "run6")}
print("== four-business held-out set (cs6_eval_human): run3 / run5 / run6")
for src in rs["run6"]["sources"]:
    print(f"{src:22s}" + " ".join(f"{rs[k]['sources'][src]['acc']:>8.3f}" for k in rs) + f"   n={rs['run6']['sources'][src]['n']}")
for key in ("accuracy", "brier", "ece"):
    print(f"overall {key:14s}" + " ".join(f"{rs[k]['overall'][key]:>8.3f}" for k in rs))
for t in ("choice", "noul", "score"):
    print(f"teacher agree {t:8s}" + " ".join(f"{rs[k]['teacher'][t]['argmax_agreement']:>8.3f}" for k in rs))
print("tickets              " + " ".join(f"{rs[k]['tickets']['department']}/5 {rs[k]['tickets']['refund']}/5 {rs[k]['tickets']['frustration_mae']:.2f}".rjust(8) for k in rs))
print()
print("== public set: run3 / run5 / run6")
p = {k: R(k) for k in ("run3", "run5", "run6")}
for src in p["run6"]["sources"]:
    print(f"{src:22s}" + " ".join(f"{p[k]['sources'][src]['acc']:>8.3f}" for k in p))
for key in ("accuracy", "brier", "ece"):
    print(f"overall {key:14s}" + " ".join(f"{p[k]['overall'][key]:>8.3f}" for k in p))
m = json.load(open(os.path.join(here, "cs6_manifest.json"), encoding="utf-8"))
print()
print("items", m["items"], m["by_type"], "eval records", m["eval_records"], "by_source", m["by_source"])
