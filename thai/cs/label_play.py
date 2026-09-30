"""Google Play reviews -> run 12 items. Two labellers, kept only where they agree on the intent: the teacher (all questions,
soft targets) and the current student (--student, intent). Agreement trades recall for precision; the teacher alone is 31%
right on real text and the student alone would learn its own mistakes. Frustration is not taken from the teacher (it
over-scores). Replay: a sample of the run 11 items (real long + short posts, human frustration, synthetic).

    python label_play.py --teacher http://172.18.72.145:8010 --student /work/thai/out/laya-th-run11
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import laya  # noqa: E402
from cs_questions import question_set  # noqa: E402
from distill_from_ots import ask_teacher, targets_from  # noqa: E402
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402

QIDS = ["intent", "intent", "business", "department", "urgency", "wants_human", "wants_refund", "has_reference", "sentiment"]


def one_hot(q, label, smooth):
    keys = list(q["criteria"])
    v = [smooth / len(keys)] * len(keys)
    v[keys.index(label)] += 1.0 - smooth
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/work/thai/data_domain/play_reviews.jsonl")
    ap.add_argument("--out", default="/work/thai/data/cs")
    ap.add_argument("--prefix", default="cs12")
    ap.add_argument("--teacher", default="http://172.18.72.145:8010")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run11")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--replay", default="/work/thai/data/cs/cs11_items.pt")
    ap.add_argument("--replay-n", type=int, default=40000)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--min-teacher-p", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    t0 = time.perf_counter()
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    if args.limit:
        rng.shuffle(rows)
        rows = rows[: args.limit]
    for r in rows:
        r["questions"] = question_set(r["business"])
    print(f"{len(rows)} reviews {dict(Counter(r['business'] for r in rows))}", flush=True)

    # ---- student intent
    agent = laya.Agent(args.student, device="cuda")
    for i, r in enumerate(rows):
        r["student_intent"] = agent.predict(r["text"], {"intent": r["questions"]["intent"]})["answers"]["intent"]["choice"]
        if (i + 1) % 2000 == 0:
            print(f"  student {i + 1}/{len(rows)}, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    del agent

    # ---- teacher, all questions
    done = [0]

    def work(r):
        res = ask_teacher(args.teacher, r["text"], r["questions"])
        done[0] += 1
        if done[0] % 2000 == 0:
            print(f"  teacher {done[0]}/{len(rows)}, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if res is None:
            return None
        t = targets_from(res["answers"], r["questions"])
        keys = list(r["questions"]["intent"]["criteria"])
        return {**r, "targets": t, "teacher_intent": keys[max(range(len(keys)), key=t["intent"].__getitem__)]}

    with ThreadPoolExecutor(args.workers) as ex:
        labelled = [x for x in ex.map(work, rows) if x]
    # agreement: the student's intent, when the teacher gives it at least --min-teacher-p (top-1 agreement is only ~15%: the
    # teacher spreads probability over neighbouring intents on these short reviews)
    strict = sum(r["teacher_intent"] == r["student_intent"] for r in labelled)
    agree = []
    for r in labelled:
        keys = list(r["questions"]["intent"]["criteria"])
        if r["targets"]["intent"][keys.index(r["student_intent"])] >= args.min_teacher_p:
            r["teacher_intent"] = r["student_intent"]
            agree.append(r)
    print(f"teacher answered {len(labelled)}; strict intent agreement {strict}/{len(labelled)} = {strict / max(1, len(labelled)):.3f}; "
          f"kept (teacher p(student intent) >= {args.min_teacher_p}) {len(agree)} = {len(agree) / max(1, len(labelled)):.3f}; "
          f"intents {dict(Counter(r['teacher_intent'] for r in agree).most_common(12))}", flush=True)
    with open(Path(args.out) / f"{args.prefix}_play.jsonl", "w", encoding="utf-8") as f:
        for r in labelled:
            f.write(json.dumps({k: v for k, v in r.items() if k != "questions"}, ensure_ascii=False) + "\n")

    # ---- items: agreed rows only; intent one-hot on the agreed label, business one-hot unless other, the rest the teacher's
    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    items, dropped = [], 0
    for r in agree:
        q = r["questions"]
        r["targets"]["intent"] = one_hot(q["intent"], r["teacher_intent"], args.smooth)
        if r["teacher_intent"] != "other":
            r["targets"]["business"] = one_hot(q["business"], r["business"], args.smooth)
        for qid in QIDS:
            qq = q[qid]
            qi = {"t": qq["type"], "ins": qq["instructions"], "crit": qq.get("criteria") or ({} if qq["type"] == "noul" else None)}
            seq, markers = build_sequence(tok, r["text"], qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                dropped += 1
                continue
            t = r["targets"][qid]
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[qq["type"]], "target": t,
                          "label": max(range(len(t)), key=t.__getitem__), "source": f"play_{r['business']}"})
    n_play = len(items)
    pool = torch.load(args.replay, weights_only=True)
    items += rng.sample(pool, min(args.replay_n, len(pool)))
    del pool
    rng.shuffle(items)
    torch.save(items, Path(args.out) / f"{args.prefix}_items.pt")
    summary = {"reviews": len(rows), "teacher_answered": len(labelled), "agreed": len(agree), "play_items": n_play, "dropped": dropped,
               "items": len(items), "replay": len(items) - n_play, "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (Path(args.out) / f"{args.prefix}_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
