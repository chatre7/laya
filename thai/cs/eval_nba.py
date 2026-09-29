"""Next-best-action eval on the 120 real banking rows x 4 contexts (label_nba.py -> nba_eval_banking.jsonl).

Systems: laya checkpoints asked NBA_Q directly; the teacher asked NBA_Q directly; and the rule baseline = a laya intent
answer (cs_questions intent question) -> nba_actions.INTENT_DEFAULT -> apply_context, i.e. what a call center gets from
the existing intent model plus a lookup table. top-1 counts when the answer is the best or an acceptable action; top-3 when
the best action is among the three highest.

    python eval_nba.py --models /work/thai/out/laya-th-nba1,/work/thai/out/laya-th-run8 --rule-model /work/thai/out/laya-th-run8 \
        --teacher http://172.18.72.145:8010 --out /work/thai/out/nba1_eval.json
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402
from cs_questions import intent_question  # noqa: E402
from distill_from_ots import ask_teacher  # noqa: E402
from nba_actions import INTENT_DEFAULT, NBA_BASE_Q, apply_context  # noqa: E402


def ranked(probs):
    return [k for k, _ in sorted(probs.items(), key=lambda kv: -kv[1])]


def context_of(r):
    v, h = r["source"].split("_")[1:]
    return v == "v1", int(h[1:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="/work/thai/data/nba/nba_eval_banking.jsonl")
    ap.add_argument("--models", default="/work/thai/out/laya-th-nba1,/work/thai/out/laya-th-run8")
    ap.add_argument("--rule-model", default="/work/thai/out/laya-th-run8")
    ap.add_argument("--teacher", default="http://172.18.72.145:8010")
    ap.add_argument("--out", default="/work/thai/out/nba1_eval.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.eval, encoding="utf-8")]
    preds = {}  # system -> list of ranked action lists

    for m in [x for x in args.models.split(",") if x]:
        agent = laya.Agent(m, device="cuda")
        preds[os.path.basename(m)] = [ranked(agent.predict(r["state"], r["questions"])["answers"]["next_action"]["probabilities"]) for r in rows]
        del agent
    if args.teacher:
        preds["teacher (asked with context)"], preds["teacher (message) + playbook"], base = [], [], {}
        for r in rows:
            res = ask_teacher(args.teacher, r["state"], r["questions"])
            preds["teacher (asked with context)"].append(ranked(res["answers"]["next_action"]["probabilities"]) if res else [])
            msg = r["state"]["ข้อความลูกค้า"]
            if msg not in base:  # the labelling setup: message only, base context, then the playbook rule
                res = ask_teacher(args.teacher, msg, {"next_action": NBA_BASE_Q})
                base[msg] = ranked(res["answers"]["next_action"]["probabilities"]) if res else []
            v, h = context_of(r)
            acts = []
            for a in base[msg]:
                a = apply_context(a, v, h)
                if a not in acts:
                    acts.append(a)
            preds["teacher (message) + playbook"].append(acts)
    if args.rule_model:
        agent = laya.Agent(args.rule_model, device="cuda")
        iq = {"intent": intent_question("banking")}
        cache, out = {}, []
        for r in rows:
            msg = r["state"]["ข้อความลูกค้า"]
            if msg not in cache:
                cache[msg] = ranked(agent.predict(msg, iq)["answers"]["intent"]["probabilities"])
            v, h = context_of(r)
            acts = []
            for intent in cache[msg]:
                a = apply_context(INTENT_DEFAULT.get(intent, "out_of_scope"), v, h)
                if a not in acts:
                    acts.append(a)
            out.append(acts)
        preds["rule(" + os.path.basename(args.rule_model) + " intent -> table)"] = out
        del agent

    summary = {}
    print(f"{len(rows)} decisions ({len(rows) // 4} real messages x 4 contexts)")
    print(f"{'system':44s} {'top-1':>6s} {'top-3':>6s}   top-1 by context (verified&first / unverified / 3rd contact)")
    for name, ps in preds.items():
        by = defaultdict(lambda: [0, 0])
        t1 = t3 = 0
        wrong = Counter()
        for r, p in zip(rows, ps):
            ok = bool(p) and p[0] in r["accept"]
            t1 += ok
            t3 += r["labels"]["next_action"] in p[:3]
            v, h = context_of(r)
            ctx = "base" if v and h == 1 else ("repeat" if h == 3 else "unverified")
            by[ctx][0] += ok
            by[ctx][1] += 1
            if not ok and ctx == "base" and p:
                wrong[f'{r["labels"]["next_action"]} -> {p[0]}'] += 1
        summary[name] = {"top1": round(t1 / len(rows), 4), "top3": round(t3 / len(rows), 4),
                         "by_context": {k: round(a / b, 4) for k, (a, b) in by.items()}, "common_errors_base": wrong.most_common(8)}
        bc = summary[name]["by_context"]
        print(f"{name[:44]:44s} {t1 / len(rows):6.3f} {t3 / len(rows):6.3f}   {bc.get('base', 0):.3f} / {bc.get('unverified', 0):.3f} / {bc.get('repeat', 0):.3f}")
    for name, s in summary.items():
        print(f"\n{name} common errors (base context, gold -> answer): {s['common_errors_base']}")
    json.dump(summary, open(args.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
