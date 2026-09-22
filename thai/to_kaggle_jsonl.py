"""Convert our records to the {state, questions, gold} JSONL that the Kaggle notebook
"Fine-tune Laya (RLCD) on your own typed decisions" reads with DATA_SOURCE = "jsonl".

    python to_kaggle_jsonl.py --human data/eval.jsonl --distill data/distill.jsonl --out kaggle/train.jsonl

gold[qid]["probabilities"] is keyed like laya's own outputs: option names for choice, "0".."k-1" for
score, "false"/"true" for noul. Human labels become one-hot; distillation targets are the teacher's
probabilities as they are.
"""
import argparse
import json


def gold_from_label(q, label):
    t = q["type"]
    if t == "choice":
        keys = list(q["criteria"])
        return None if label not in keys else {"probabilities": {k: 1.0 if k == label else 0.0 for k in keys}, "choice": label}
    if t == "score":
        k = len(q["criteria"])
        return {"probabilities": {str(i): 1.0 if i == int(label) else 0.0 for i in range(k)}, "score": int(label)}
    return {"probabilities": {"false": 0.0 if label else 1.0, "true": 1.0 if label else 0.0}, "noul": bool(label)}


def gold_from_target(q, target):
    t = q["type"]
    if t == "choice":
        keys = list(q["criteria"])
        p = dict(zip(keys, target))
        return {"probabilities": p, "choice": max(p, key=p.get)}
    if t == "score":
        p = {str(i): v for i, v in enumerate(target)}
        return {"probabilities": p, "score": sum(i * v for i, v in enumerate(target))}
    return {"probabilities": {"false": target[0], "true": target[1]}, "noul": target[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", nargs="*", default=[], help="records with labels (prep_thai.py output)")
    ap.add_argument("--distill", nargs="*", default=[], help="records with targets (distill_from_ots.py output)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    n = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for path in args.human:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                gold = {qid: gold_from_label(q, r["labels"][qid]) for qid, q in r["questions"].items() if qid in r["labels"]}
                gold = {k: v for k, v in gold.items() if v}
                if gold:
                    f.write(json.dumps({"id": r["id"], "source": r["source"], "state": r["state"], "questions": r["questions"], "gold": gold}, ensure_ascii=False) + "\n")
                    n += 1
        for path in args.distill:
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                gold = {qid: gold_from_target(q, r["targets"][qid]) for qid, q in r["questions"].items()}
                f.write(json.dumps({"id": r["id"], "source": r["source"], "state": r["state"], "questions": r["questions"], "gold": gold}, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} cases to {args.out}")


if __name__ == "__main__":
    main()
