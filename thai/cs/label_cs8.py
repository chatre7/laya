"""Run 8 data: run 7 records (extended intent lists) + real-question sets + hand-labelled real Thai (Pantip).

Sources:
  - --from-records cs7: the run 7 labelled records (cs7.jsonl + cs7_eval.jsonl). Their business/intent targets are rebuilt as
    one-hot over the RUN 8 intent lists (cs_questions.py grew: telecom +5, banking +19, insurance +1); teacher targets for the
    shared questions are kept. E-commerce rows are capped (--ecom-cap) so the new sources are not drowned.
  - banking77_th.jsonl (19,975 Thai rewrites of real banking questions; 77 labels -> BANKING77_TO_CS) and insuranceqa_th.jsonl
    (8,758 rewrites of real insurance questions; topic -> information_*). The teacher answers all questions; a row is kept when
    the teacher puts p >= --min-p on the mapped intent or ranks it top-3 (the mapping is coarse), then the intent target is the
    human one-hot and the shared questions are the teacher's.
  - data/domain/pantip_<biz>_extra.jsonl + labels_<biz>_extra.json: 932 real Pantip questions I labelled by hand
    (intent, department, urgency). Business is one-hot when the intent is in-scope, the teacher's when it is `other`. These are
    the only real Thai in-domain texts, so their labelled questions are repeated --pantip-repeat times in the items.
  - the first 360 hand-labelled Pantip rows (real_cs_eval.jsonl) stay out of training: they are the real-text eval.

    python label_cs8.py --from-records cs7 --teacher http://HOST:8010 --workers 8 --prefix cs8
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
from cs_questions import BANKING77_TO_CS, INTENTS, ORDER, question_set  # noqa: E402
from distill_from_ots import ask_teacher, targets_from  # noqa: E402
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402


def one_hot(q, label, smooth=0.0):
    keys = list(range(len(q["criteria"]))) if q["type"] == "score" else list(q["criteria"])
    k = len(keys)
    v = [smooth / k] * k
    v[keys.index(label)] += 1.0 - smooth
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-records", default="cs7")
    ap.add_argument("--out", default="/work/thai/data/cs")
    ap.add_argument("--prefix", default="cs8")
    ap.add_argument("--domain", default="/work/thai/data/domain")
    ap.add_argument("--b77", default="/work/thai/data/cs2/banking77_th.jsonl")
    ap.add_argument("--iqa", default="/work/thai/data/cs2/insuranceqa_th.jsonl")
    ap.add_argument("--teacher", default="http://localhost:8010")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-p", type=float, default=0.3, help="keep banking77/insurance-qa rows whose teacher p(mapped intent) >= this or top-3")
    ap.add_argument("--ecom-cap", type=int, default=30000)
    ap.add_argument("--pantip-repeat", type=int, default=6, help="repeat each hand-labelled Pantip question this many times in the items")
    ap.add_argument("--questions-per-record", type=int, default=3)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--student", default="/work/thai/out/laya-th-run3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: records per source (prior records too)")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out)
    domain = Path(args.domain)
    teachers = args.teacher.split(",")
    t0 = time.perf_counter()

    # ---- run 7 records, targets rebuilt over the run 8 intent lists
    prior, n_ecom, by_prior = [], 0, Counter()
    for name in (f"{args.from_records}.jsonl", f"{args.from_records}_eval.jsonl"):
        for line in open(out / name, encoding="utf-8"):
            r = json.loads(line)
            biz = r["labels"]["business"]
            if biz == "ecommerce":
                n_ecom += 1
                if args.ecom_cap and n_ecom > args.ecom_cap:
                    continue
            if args.limit and by_prior[r["source"]] >= args.limit:
                continue
            by_prior[r["source"]] += 1
            r["questions"] = question_set(biz if biz in INTENTS else rng.choice(sorted(INTENTS)))
            for qid, lab in r["labels"].items():
                r["targets"][qid] = one_hot(r["questions"][qid], lab, args.smooth)
            prior.append(r)
    print(f"reusing {len(prior)} records from {args.from_records} {dict(by_prior)}", flush=True)

    # ---- new records
    recs = []
    for biz in ("telecom", "banking", "insurance"):
        lab = json.load(open(domain / f"labels_{biz}_extra.json", encoding="utf-8"))["labels"]
        rows = [json.loads(l) for l in open(domain / f"pantip_{biz}_extra.jsonl", encoding="utf-8")]
        for i, row in enumerate(rows):
            if str(i) not in lab:
                continue
            intent, dept, urg = lab[str(i)]
            labels = {"intent": intent, "department": dept, "urgency": int(urg)}
            if intent != "other":
                labels["business"] = biz
            recs.append({"id": f"pantip-{biz}-{i}", "source": f"pantip_{biz}", "group": f"pantip-{biz}-{i}", "state": row["text"].strip(),
                         "style": "pantip", "labels": labels, "questions": question_set(biz), "human": True})
    for path, key in ((args.b77, "b77"), (args.iqa, "iqa")):
        for i, line in enumerate(open(path, encoding="utf-8")):
            r = json.loads(line)
            biz = r["business"]
            intent = BANKING77_TO_CS.get(r["intent"].strip()) if key == "b77" else r["intent"]
            if not intent or intent not in INTENTS[biz] or len(r["text"].strip()) < 5:
                continue
            recs.append({"id": f"{key}-{i}", "source": key, "group": f"{biz}|{r['text_en']}", "state": r["text"].strip(), "style": r.get("style", ""),
                         "labels": {"business": biz, "intent": intent}, "questions": question_set(biz), "gate": True})
    if args.limit:
        by, keep = Counter(), []
        for r in recs:
            if by[r["source"]] < args.limit:
                keep.append(r)
                by[r["source"]] += 1
        recs = keep
    print(f"{len(recs)} records to label: {dict(Counter(r['source'] for r in recs))}", flush=True)

    # ---- teacher for the shared questions (+ gate for the mapped real-question sets)
    done, gated = [0], Counter()

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
        if r.get("gate"):
            probs = targets["intent"]
            idx = list(r["questions"]["intent"]["criteria"]).index(r["labels"]["intent"])
            top3 = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)[:3]
            r["teacher_p"] = round(probs[idx], 4)
            if probs[idx] < args.min_p and idx not in top3:
                gated[r["source"]] += 1
                return None
        for qid, lab in r["labels"].items():
            targets[qid] = one_hot(r["questions"][qid], lab, args.smooth)
        return {**r, "targets": targets}

    with ThreadPoolExecutor(args.workers) as ex:
        labelled = [x for x in ex.map(work, enumerate(recs)) if x]
    print(f"teacher labelled {len(labelled)}/{len(recs)} in {(time.perf_counter() - t0) / 60:.1f} min; gated out {dict(gated)}", flush=True)
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
    items, dropped, by_type, by_src = [], 0, Counter(), Counter()
    for r in tr:
        if r.get("human"):
            qids = [q for q in r["labels"] for _ in range(args.pantip_repeat)] + [q for q in ORDER if q not in r["labels"]]
        else:
            first = rng.choice(list(r["labels"]))
            others = [q for q in ORDER if q != first]
            qids = [first] + rng.sample(others, max(0, args.questions_per_record - 1))
        for qid in qids:
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
            by_src[r["source"]] += 1
    rng.shuffle(items)
    torch.save(items, out / f"{px}_items.pt")
    summary = {"records": len(labelled), "train_records": len(tr), "eval_records": len(ev), "items": len(items), "dropped": dropped,
               "by_source": dict(Counter(r["source"] for r in labelled)), "items_by_source": dict(by_src), "by_type": dict(by_type),
               "gated_out": dict(gated), "min_p": args.min_p, "ecom_cap": args.ecom_cap, "pantip_repeat": args.pantip_repeat,
               "questions_per_record": args.questions_per_record, "smooth": args.smooth, "intents": {b: len(v) for b, v in INTENTS.items()},
               "head_max_len": cfg["head_max_len"], "max_len": cfg["max_len"], "mean_len": sum(len(i["ids"]) for i in items) / max(1, len(items)),
               "teachers": teachers, "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (out / f"{px}_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
