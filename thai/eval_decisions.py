"""Evaluate laya checkpoints on the frozen eval suites of hagsmand1/laya-thai-decisions (17 suites, 9,161 cases: Thai
MASSIVE intent, unseen instruction phrasings on MASSIVE / wisesight / iapp, unseen synthetic domains, and four probes).
Not our data and not our questions: states are often objects, instructions point at a field in backticks, option keys are
letters / Thai words / slugs, scales run in both directions.

    python eval_decisions.py --models convaiinnovations/laya-multilingual,/work/thai/out/laya-th-run16 --out /work/thai/out/decisions.json

`expected` holds the option key for choice (an index is accepted too), the level index for score, a boolean for noul.
Score is read as the most probable level. A record the model cannot take (options over the head budget) counts as wrong.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "cs"))
from agents import load_agent  # noqa: E402

REPO = "hagsmand1/laya-thai-decisions"


def answer(q, a):
    """The model's answer in the form `expected` uses."""
    if q["type"] == "choice":
        return a["choice"]
    if q["type"] == "noul":
        return a["noul"] > 0.5
    p = a["probabilities"]
    return max(range(len(q["criteria"])), key=lambda i: p[str(i)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suites", default="", help="comma-separated split names; all when empty")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    data = load_dataset(REPO, "eval")
    suites = [s for s in data if not args.suites or s in args.suites.split(",")]
    report = {}
    for path in args.models.split(","):
        name = path.rstrip("/").split("/")[-1]
        try:
            agent = load_agent(path, args.device)
        except Exception as e:  # noqa: BLE001
            print(f"== {name}: not loaded ({type(e).__name__}: {e})", flush=True)
            continue
        rep = report[name] = {}
        print(f"== {name}", flush=True)
        for suite in suites:
            n = ok = errors = 0
            by = defaultdict(lambda: [0, 0])  # question type or qid -> [correct, n]
            pairs = []  # probes: (answers, expected) per case
            for r in data[suite]:
                questions, expected = json.loads(r["questions"]), json.loads(r["expected"])
                try:
                    res = agent.predict(json.loads(r["state"]), questions)["answers"]
                except Exception:  # noqa: BLE001
                    errors += 1
                    res = None
                got = {}
                for qid, q in questions.items():
                    exp = expected[qid]
                    if q["type"] == "choice" and isinstance(exp, int):
                        exp = list(q["criteria"])[exp]
                    got[qid] = None if res is None else answer(q, res[qid])
                    hit = got[qid] is not None and got[qid] == exp
                    n += 1
                    ok += hit
                    for key in (q["type"], qid) if suite.startswith("probe_") else (q["type"],):
                        by[key][0] += hit
                        by[key][1] += 1
                pairs.append((got, expected))
            s = rep[suite] = {"cases": len(pairs), "questions": n, "accuracy": round(ok / max(1, n), 4), "errors": errors,
                              "by": {k: [c, m, round(c / m, 4)] for k, (c, m) in sorted(by.items())}}
            extra = ""
            if suite == "probe_noul_negation":  # a claim and its negation must get opposite answers
                s["flip"] = round(sum(g["claim"] is not None and g["claim"] != g["negated"] for g, _ in pairs) / len(pairs), 4)
                extra = f" flip {s['flip']:.3f}"
            elif suite == "probe_noul_labels":  # the same question with the two answers relabelled must get the same answer
                s["agree"] = round(sum(g["default"] is not None and g["default"] == g["relabelled"] for g, _ in pairs) / len(pairs), 4)
                true = [(g, e) for g, e in pairs if e["default"]]
                s["true_cases"] = [sum(g["default"] is True and g["relabelled"] is True for g, _ in true), len(true)]
                extra = f" agree {s['agree']:.3f}, both right on the true cases {s['true_cases'][0]}/{s['true_cases'][1]}"
            elif suite == "probe_score_orientation":  # the reversed scale must give the mirrored level
                k = 3
                mirrored = [g["normal"] is not None and g["reversed"] == k - 1 - g["normal"] for g, _ in pairs]
                polar = [m for m, (_, e) in zip(mirrored, pairs) if e["normal"] != 1]
                both = [g["normal"] == e["normal"] and g["reversed"] == e["reversed"] for g, e in pairs if e["normal"] != 1]
                s["mirrored"] = round(sum(mirrored) / len(pairs), 4)
                s["polar"] = {"n": len(polar), "mirrored": round(sum(polar) / max(1, len(polar)), 4), "both_right": round(sum(both) / max(1, len(both)), 4)}
                extra = f" mirrored {s['mirrored']:.3f}; non-neutral cases n={len(polar)}: mirrored {s['polar']['mirrored']:.3f}, both right {s['polar']['both_right']:.3f}"
            types = "  ".join(f"{k} {c}/{m}={v:.3f}" for k, (c, m, v) in s["by"].items())
            print(f"{suite:38s} {len(pairs):5d} cases  acc {s['accuracy']:.3f}  [{types}]{extra}" + (f"  errors {errors}" if errors else ""), flush=True)
        del agent
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
