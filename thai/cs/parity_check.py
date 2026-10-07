"""Does a checkpoint answer the same under two versions of the laya code? Run once per code tree with the same checkpoint
and texts, then compare: every answer (choice / level / yes-no) must match and the probabilities must be close. Used
before merging the upstream repository into ours (the served checkpoints were trained on the older code).

    python parity_check.py --model /work/thai/out/laya-th-run19 --out /work/thai/out/parity_old.json
    python parity_check.py --compare /work/thai/out/parity_old.json /work/thai/out/parity_new.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/work/thai/out/laya-th-run19")
    ap.add_argument("--texts", default="/work/thai/data_domain/new_check_all.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        a, b = (json.load(open(p, encoding="utf-8")) for p in args.compare)
        print(f"code: {a['laya']} vs {b['laya']}; {len(a['rows'])} texts")
        same = diff = 0
        worst = 0.0
        for ra, rb in zip(a["rows"], b["rows"]):
            for q in ra:
                x, y = ra[q], rb[q]
                if x["answer"] == y["answer"]:
                    same += 1
                else:
                    diff += 1
                    if diff <= 8:
                        print(f"  differs: {q} {x['answer']} -> {y['answer']}")
                worst = max(worst, max(abs(x["p"][k] - y["p"].get(k, 0)) for k in x["p"]))
        print(f"answers equal {same}, different {diff}; largest probability gap {worst:.4f}")
        return
    import laya
    from cs_questions import EXTRA, SHARED, intent_question
    rows = [json.loads(l) for l in open(args.texts, encoding="utf-8")][: args.n]
    agent = laya.Agent(args.model, device="cuda")
    out = []
    for r in rows:
        qs = {"intent": intent_question(r["business"]), "department": SHARED["department"], "urgency": SHARED["urgency"],
              "frustration": SHARED["frustration"], "sentiment": SHARED["sentiment"], "churn": EXTRA["churn_threat"], "effort": EXTRA["contact_effort"]}
        ans = agent.predict(r["text"], qs)["answers"]
        rec = {}
        for q, a in ans.items():
            if a["type"] == "noul":
                rec[q] = {"answer": a["noul"] > 0.5, "p": {"true": round(a["noul"], 4)}}
            elif a["type"] == "score":
                p = {k: round(v, 4) for k, v in a["probabilities"].items()}
                rec[q] = {"answer": max(p, key=p.get), "p": p}
            else:
                rec[q] = {"answer": a["choice"], "p": {k: round(v, 4) for k, v in a["probabilities"].items()}}
        out.append(rec)
    ver = getattr(laya, "__version__", "?")
    json.dump({"laya": ver, "model": args.model, "rows": out}, open(args.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"laya {ver}: {len(out)} texts -> {args.out}")


if __name__ == "__main__":
    main()
