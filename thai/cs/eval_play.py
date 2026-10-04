"""Score checkpoints on the hand-labelled Google Play reviews (data_domain/play_labels.json: id -> [intent, department, urgency]
over play_reviews_eval_sample.jsonl): intent / department / urgency accuracy and false-other, per business.

    python eval_play.py --models /work/thai/out/laya-th-run11,/work/thai/out/laya-th-run12 --out /work/thai/out/play12.json
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from agents import load_agent  # noqa: E402
from cs_questions import SHARED, intent_question  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--sample", default="/work/thai/data_domain/play_reviews_eval_sample.jsonl")
    ap.add_argument("--labels", default="/work/thai/data_domain/play_labels.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    labels = json.load(open(args.labels, encoding="utf-8"))["labels"]
    rows = [json.loads(l) for l in open(args.sample, encoding="utf-8")]
    rows = [r for r in rows if r["id"] in labels]
    report = {}
    for m in args.models.split(","):
        agent = load_agent(m)
        st = defaultdict(lambda: defaultdict(float))
        for r in rows:
            intent, dept, urg = labels[r["id"]]
            a = agent.predict(r["text"], {"intent": intent_question(r["business"]), "department": SHARED["department"], "urgency": SHARED["urgency"]})["answers"]
            for key in (r["business"], "all"):
                s = st[key]
                s["n"] += 1
                s["intent"] += a["intent"]["choice"] == intent
                s["department"] += a["department"]["choice"] == dept
                s["urgency"] += round(a["urgency"]["score"]) == int(urg)
                if intent == "other":
                    s["n_other"] += 1
                    s["other_caught"] += a["intent"]["choice"] == "other"
                else:
                    s["n_in"] += 1
                    s["false_other"] += a["intent"]["choice"] == "other"
        del agent
        name = os.path.basename(m)
        report[name] = {}
        print(f"\n== {name}  ({len(rows)} hand-labelled reviews)")
        for key in ("telecom", "banking", "insurance", "all"):
            s = st[key]
            if not s["n"]:
                continue
            line = {"n": int(s["n"]), "intent": round(s["intent"] / s["n"], 4), "department": round(s["department"] / s["n"], 4),
                    "urgency": round(s["urgency"] / s["n"], 4), "false_other": round(s["false_other"] / max(1, s["n_in"]), 4),
                    "other_caught": round(s["other_caught"] / max(1, s["n_other"]), 4)}
            report[name][key] = line
            print(f"{key:10s} n={line['n']:4d} intent {line['intent']:.3f} dept {line['department']:.3f} urg {line['urgency']:.3f} "
                  f"false-other {int(s['false_other'])}/{int(s['n_in'])} other-caught {int(s['other_caught'])}/{int(s['n_other'])}")
    json.dump(report, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
