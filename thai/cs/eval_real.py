"""Real-text eval, local checkpoints: the 360 hand-labelled Pantip posts (long) and their short chat versions (short,
shorten_pantip.py set "eval"), per business: intent, department, urgency exact, how often an in-scope message is called
`other` (false-other) and how often an `other` one is caught; on the long posts also frustration against the hand labels
(data_domain/frustration_real_eval.json): exact, MAE and the mean predicted vs the mean true score (over-scoring shows here).

    python eval_real.py --models /work/thai/out/laya-th-run8,/work/thai/out/laya-th-run11 --out /work/thai/out/real11.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402
from cs_questions import SHARED, intent_question  # noqa: E402

BIZ = ("telecom", "banking", "insurance")


def load(args):
    long_rows = [json.loads(l) for l in open(args.long, encoding="utf-8")]
    frus = json.load(open(args.frustration, encoding="utf-8"))["labels"]
    rows = [{"set": "long", "business": r["source"], "id": r["id"], "text": r["state"], "labels": r["labels"], "frustration": frus.get(r["id"])}
            for r in long_rows]
    for l in open(args.short, encoding="utf-8"):
        s = json.loads(l)
        if s["set"] == "eval":
            rows.append({"set": "short", "business": s["business"], "id": f'{s["id"]}-s{s["variant"]}', "text": s["text"], "labels": s["labels"],
                         "frustration": None})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--long", default="/work/thai/data/domain/real_cs_eval.jsonl")
    ap.add_argument("--short", default="/work/thai/data/cs/pantip_short.jsonl")
    ap.add_argument("--frustration", default="/work/thai/data_domain/frustration_real_eval.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = load(args)
    report = {}
    for m in args.models.split(","):
        agent = laya.Agent(m, device="cuda")
        st = defaultdict(lambda: defaultdict(float))
        for r in rows:
            qs = {"intent": intent_question(r["business"]), "department": SHARED["department"], "urgency": SHARED["urgency"]}
            if r["frustration"] is not None:
                qs["frustration"] = SHARED["frustration"]
            a = agent.predict(r["text"], qs)["answers"]
            lab = r["labels"]
            s = st[(r["set"], r["business"])]
            s["n"] += 1
            s["intent"] += a["intent"]["choice"] == lab["intent"]
            s["department"] += a["department"]["choice"] == lab["department"]
            s["urgency"] += round(a["urgency"]["score"]) == lab["urgency"]
            if lab["intent"] == "other":
                s["n_other"] += 1
                s["other_caught"] += a["intent"]["choice"] == "other"
            else:
                s["n_in"] += 1
                s["false_other"] += a["intent"]["choice"] == "other"
            if r["frustration"] is not None:
                f = a["frustration"]["score"]
                s["n_f"] += 1
                s["f_exact"] += round(f) == r["frustration"]
                s["f_abs"] += abs(f - r["frustration"])
                s["f_pred"] += f
                s["f_true"] += r["frustration"]
        del agent
        name = os.path.basename(m)
        report[name] = {}
        print(f"\n== {name}")
        print(f"{'set':6s} {'business':10s} {'n':>4s} {'intent':>7s} {'dept':>6s} {'urg':>6s} {'false-other':>12s} {'other-caught':>13s} | frustration exact / MAE / mean pred vs true")
        for sset in ("long", "short"):
            tot = defaultdict(float)
            for biz in BIZ:
                s = st[(sset, biz)]
                if not s["n"]:
                    continue
                for k, v in s.items():
                    tot[k] += v
                line = {"n": int(s["n"]), "intent": s["intent"] / s["n"], "department": s["department"] / s["n"], "urgency": s["urgency"] / s["n"],
                        "false_other": s["false_other"] / max(1, s["n_in"]), "other_caught": s["other_caught"] / max(1, s["n_other"])}
                if s["n_f"]:
                    line.update({"frustration_exact": s["f_exact"] / s["n_f"], "frustration_mae": s["f_abs"] / s["n_f"],
                                 "frustration_mean_pred": s["f_pred"] / s["n_f"], "frustration_mean_true": s["f_true"] / s["n_f"]})
                report[name][f"{sset}_{biz}"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in line.items()}
                fr = (f"{line['frustration_exact']:.3f} / {line['frustration_mae']:.2f} / {line['frustration_mean_pred']:.2f} vs {line['frustration_mean_true']:.2f}"
                      if s["n_f"] else "")
                print(f"{sset:6s} {biz:10s} {int(s['n']):4d} {line['intent']:7.3f} {line['department']:6.3f} {line['urgency']:6.3f} "
                      f"{int(s['false_other']):5d}/{int(s['n_in']):<5d} {int(s['other_caught']):6d}/{int(s['n_other']):<5d} | {fr}")
            if tot["n"]:
                report[name][f"{sset}_all"] = {"n": int(tot["n"]), "intent": round(tot["intent"] / tot["n"], 4), "department": round(tot["department"] / tot["n"], 4),
                                               "urgency": round(tot["urgency"] / tot["n"], 4), "false_other": round(tot["false_other"] / max(1, tot["n_in"]), 4)}
                a = report[name][f"{sset}_all"]
                print(f"{sset:6s} {'all':10s} {a['n']:4d} {a['intent']:7.3f} {a['department']:6.3f} {a['urgency']:6.3f} {a['false_other']:12.3f}")
    json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
