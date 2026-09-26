"""Run 6 data: four businesses (telecom, banking, insurance, e-commerce) + out-of-scope, the cs question set.

Records:
  - thai/data/cs/cs_colloquial_checked.jsonl  (telecom/banking/insurance rewrites; keep teacher_p >= --min-p)  labels: business, intent
  - thai/data/cc/porameht_colloquial_kept.jsonl (e-commerce rewrites; Bitext intent mapped with BITEXT_TO_ECOM)  labels: business, intent
  - thai/data/cc/wisesight_train.jsonl (--wisesight real Thai posts)                                            labels: business=other, intent=other
Each record carries its own `questions` (business + that business's intent list + shared). Human labels become smoothed one-hot
targets, the teacher's probabilities fill the shared questions (all questions in one request, two teachers round-robin).
Items: --questions-per-record per record = one human-labelled question at random + sampled others.
Eval split 5% by source sentence -> cs6_eval.jsonl (targets), cs6_eval_human.jsonl (eval_thai.py format).

    python label_cs.py --teacher http://HOST:8010,http://HOST:8013 --workers 16
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cs_questions import BITEXT_TO_ECOM, INTENTS, ORDER, question_set  # noqa: E402
from distill_from_ots import ask_teacher, targets_from  # noqa: E402
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402

WISESIGHT_SENT = {0: "positive", 1: "neutral", 2: "negative", 3: "question"}


def one_hot(q, label, smooth=0.0):
    keys = list(q["criteria"])
    k = len(keys)
    v = [smooth / k] * k
    v[keys.index(label)] += 1.0 - smooth
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs", default="/work/thai/data/cs/cs_colloquial_checked.jsonl")
    ap.add_argument("--ecom", default="/work/thai/data/cc/porameht_colloquial_kept.jsonl")
    ap.add_argument("--wisesight-file", default="/work/thai/data/cc/wisesight_train.jsonl")
    ap.add_argument("--out", default="/work/thai/data/cs")
    ap.add_argument("--prefix", default="cs6")
    ap.add_argument("--teacher", default="http://localhost:8010")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--min-p", type=float, default=0.5, help="keep cs rewrites whose teacher p(source intent) >= this")
    ap.add_argument("--wisesight", type=int, default=12000)
    ap.add_argument("--other-file", default="", help="in-register out-of-scope texts (gen_other.sh output): business=other, intent=other")
    ap.add_argument("--ecom-cap", type=int, default=0, help="cap e-commerce rows (0 = all)")
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--questions-per-record", type=int, default=3)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--student", default="/work/thai/out/laya-th-run3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: records per source")
    ap.add_argument("--from-records", default="", help="reuse the labelled records of an earlier prefix (e.g. cs6): only --other-file rows go to the teacher")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    teachers = args.teacher.split(",")

    # ---- records
    recs = []
    for line in open(args.cs, encoding="utf-8"):
        r = json.loads(line)
        if r.get("teacher_p", 0) < args.min_p or r["intent"] not in INTENTS[r["business"]]:
            continue
        recs.append({"id": f"{r['business']}-{len(recs)}", "source": r["business"], "group": r["business"] + "|" + r["text_en"], "state": r["text"],
                     "style": r.get("style", ""), "labels": {"business": r["business"], "intent": r["intent"]}})
    n_cs = len(recs)
    ecom = [json.loads(l) for l in open(args.ecom, encoding="utf-8")]
    rng.shuffle(ecom)
    if args.ecom_cap:
        ecom = ecom[: args.ecom_cap]
    for r in ecom:
        intent = BITEXT_TO_ECOM.get(r["intent"].strip())
        if not intent:
            continue
        recs.append({"id": f"ecommerce-{len(recs)}", "source": "ecommerce", "group": "ecommerce|" + r["src"], "state": r["text"],
                     "style": r.get("style", ""), "labels": {"business": "ecommerce", "intent": intent}})
    ws = [json.loads(l) for l in open(args.wisesight_file, encoding="utf-8")]
    ws = [r for r in ws if len(r["text"].strip()) >= 10]
    rng.shuffle(ws)
    for i, r in enumerate(ws[: args.wisesight]):
        recs.append({"id": f"other-{i}", "source": "wisesight", "group": f"ws-{i}", "state": r["text"].strip(), "style": "",
                     "labels": {"business": "other", "intent": "other", "sentiment": WISESIGHT_SENT[int(r["label"])]}})
    if args.other_file:
        for i, line in enumerate(open(args.other_file, encoding="utf-8")):
            r = json.loads(line)
            recs.append({"id": f"other-gen-{i}", "source": "other_gen", "group": f"og-{i}", "state": r["text"].strip(), "style": r.get("style", ""),
                         "labels": {"business": "other", "intent": "other"}})
    if args.limit:
        by = Counter()
        keep = []
        for r in recs:
            if by[r["source"]] < args.limit:
                keep.append(r)
                by[r["source"]] += 1
        recs = keep
    prior = []
    if args.from_records:  # reuse earlier labelled records (targets included); only the new out-of-scope rows go to the teacher
        recs = [r for r in recs if r["source"] == "other_gen"]
        by_len = {len(INTENTS[b]) + 1: b for b in INTENTS}  # intent target length -> business (telecom/banking both 27: either list is valid)
        for name in (f"{args.from_records}.jsonl", f"{args.from_records}_eval.jsonl"):
            for line in open(out / name, encoding="utf-8"):
                r = json.loads(line)
                biz = r["labels"]["business"]
                r["questions"] = question_set(biz if biz in INTENTS else by_len.get(len(r["targets"]["intent"]), rng.choice(sorted(INTENTS))))
                prior.append(r)
        print(f"reusing {len(prior)} labelled records from {args.from_records}", flush=True)
    for r in recs:
        biz = r["labels"]["business"]
        r["questions"] = question_set(biz if biz in INTENTS else rng.choice(sorted(INTENTS)))  # out-of-scope texts get a random business's intent list
    print(f"{len(recs)} records to label: cs kept {n_cs}, by source {dict(Counter(r['source'] for r in recs))}", flush=True)

    # ---- teacher for the shared questions
    t0 = time.perf_counter()
    done = [0]

    def work(ir):
        i, r = ir
        res = ask_teacher(teachers[i % len(teachers)], r["state"], r["questions"])
        done[0] += 1
        if done[0] % 2000 == 0:
            el = time.perf_counter() - t0
            print(f"  {done[0]}/{len(recs)} labelled, {el / 60:.1f} min, {done[0] / el:.1f} rec/s", flush=True)
        if res is None:
            return None
        targets = targets_from(res["answers"], r["questions"])
        for qid, lab in r["labels"].items():
            targets[qid] = one_hot(r["questions"][qid], lab, args.smooth)
        return {**r, "targets": targets}

    with ThreadPoolExecutor(args.workers) as ex:
        labelled = [x for x in ex.map(work, enumerate(recs)) if x]
    print(f"teacher labelled {len(labelled)}/{len(recs)} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    labelled = prior + labelled

    # ---- split by source sentence
    groups = sorted({r["group"] for r in labelled})
    rng.shuffle(groups)
    eval_groups = set(groups[: int(len(groups) * args.eval_frac)])
    ev = [r for r in labelled if r["group"] in eval_groups]
    tr = [r for r in labelled if r["group"] not in eval_groups]
    rng.shuffle(tr)
    px = args.prefix
    with open(out / f"{px}.jsonl", "w", encoding="utf-8") as f:
        for r in tr:
            f.write(json.dumps({k: v for k, v in r.items() if k != "questions"}, ensure_ascii=False) + "\n")
    with open(out / f"{px}_eval.jsonl", "w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / f"{px}_eval_human.jsonl", "w", encoding="utf-8") as f:
        for r in ev:
            f.write(json.dumps({"id": r["id"], "source": r["source"], "state": r["state"],
                                "questions": {q: r["questions"][q] for q in r["labels"]}, "labels": r["labels"]}, ensure_ascii=False) + "\n")

    # ---- items
    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    items, dropped, by_type = [], 0, Counter()
    for r in tr:
        first = rng.choice(list(r["labels"]))
        others = [q for q in ORDER if q != first]
        for qid in [first] + rng.sample(others, max(0, args.questions_per_record - 1)):
            q = r["questions"][qid]
            qi = {"t": q["type"], "ins": q["instructions"], "crit": q.get("criteria") or ({} if q["type"] == "noul" else None)}
            seq, markers = build_sequence(tok, r["state"], qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                dropped += 1
                continue
            target = r["targets"][qid]
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]], "target": target,
                          "label": max(range(len(target)), key=target.__getitem__), "source": r["source"]})
            by_type[q["type"]] += 1
    rng.shuffle(items)
    torch.save(items, out / f"{px}_items.pt")
    summary = {"records": len(labelled), "train_records": len(tr), "eval_records": len(ev), "items": len(items), "dropped": dropped,
               "by_source": dict(Counter(r["source"] for r in labelled)), "by_type": dict(by_type), "questions_per_record": args.questions_per_record,
               "smooth": args.smooth, "min_p": args.min_p, "head_max_len": cfg["head_max_len"], "max_len": cfg["max_len"],
               "mean_len": sum(len(i["ids"]) for i in items) / max(1, len(items)), "teachers": teachers, "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (out / f"{px}_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
