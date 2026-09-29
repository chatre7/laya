"""Per-question Mind2Web accuracy (element = target, operation, both right = step) for one laya model, plus the random-element
and always-CLICK baselines. eval_thai.py pools the two questions into one number; this splits them.

    python eval_web.py --model /work/thai/out/laya-th-web1 --out /work/thai/out/web1_m2w_split.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="/work/thai/data/web")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    agent = laya.Agent(args.model, device="cuda")
    res = {}
    for split in ("test_task", "test_website", "test_domain"):
        for lang in ("th", "en"):
            n = el = op = step = rnd = click = 0
            for line in open(Path(args.data) / f"web_eval_{split}_{lang}.jsonl", encoding="utf-8"):
                r = json.loads(line)
                try:
                    a = agent.predict(r["state"], r["questions"])["answers"]
                except Exception:  # noqa: BLE001
                    continue
                n += 1
                e = a["target"]["choice"] == r["labels"]["target"]
                o = a["operation"]["choice"] == r["labels"]["operation"]
                el += e
                op += o
                step += e and o
                rnd += 1 / len(r["questions"]["target"]["criteria"])
                click += r["labels"]["operation"] == "CLICK"
            res[f"{split}_{lang}"] = {"n": n, "element": round(el / n, 4), "operation": round(op / n, 4), "step": round(step / n, 4),
                                      "random_element": round(rnd / n, 4), "always_click": round(click / n, 4)}
            print(split, lang, res[f"{split}_{lang}"], flush=True)
    json.dump({"model": args.model, "results": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
