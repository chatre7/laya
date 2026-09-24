"""Run 4 data: the call-center question set, teacher-labelled on the colloquial customer-support rewrites + wisesight.

Inputs (thai/data/cc/): porameht_colloquial_kept.jsonl (text, intent, category, style, src) and wisesight_train.jsonl (text, label).
Every text gets the same 9 questions in one teacher request. Human labels (intent, category for Bitext rows; sentiment
for wisesight rows) become one-hot targets, everything else takes the teacher's probabilities as soft targets.

Outputs: cc.jsonl / cc_eval.jsonl (records with targets; eval split is grouped by source sentence so no paraphrase of an
eval sentence is in train), cc_eval_human.jsonl (eval_thai.py format, human labels only), cc_items.pt (train_single.py
format, --questions-per-record items per record: the human-labelled question + sampled teacher questions), cc_manifest.json.

    python label_cc.py --teacher http://localhost:8010,http://localhost:8013 --workers 16
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

from check_rewrites import INTENTS
from distill_from_ots import ask_teacher, targets_from
from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence, render_options

CATEGORIES = {
    "ORDER": "สั่งซื้อ/แก้ไขคำสั่งซื้อ", "SHIPPING": "ที่อยู่จัดส่ง", "CANCEL": "ยกเลิก ค่าธรรมเนียมยกเลิก", "INVOICE": "ใบแจ้งหนี้",
    "PAYMENT": "การชำระเงิน", "REFUND": "การคืนเงิน", "FEEDBACK": "ร้องเรียน รีวิว", "CONTACT": "ช่องทางติดต่อ ขอคุยกับคน",
    "ACCOUNT": "บัญชีผู้ใช้ รหัสผ่าน สมัครสมาชิก", "DELIVERY": "ตัวเลือกและระยะเวลาการจัดส่ง", "SUBSCRIPTION": "จดหมายข่าว การสมัครรับข้อมูล",
}
QUESTIONS = {
    "intent": {"type": "choice", "instructions": "ลูกค้าต้องการอะไร", "criteria": INTENTS},
    "category": {"type": "choice", "instructions": "เรื่องที่ลูกค้าติดต่ออยู่ในหมวดใด", "criteria": CATEGORIES},
    "department": {"type": "choice", "instructions": "ควรส่งเรื่องนี้ให้ทีมใดรับผิดชอบ",
                   "criteria": {"billing": "ค่าบริการ ใบแจ้งหนี้ การชำระเงิน การคืนเงิน", "shipping": "การจัดส่ง ที่อยู่ ติดตามพัสดุ",
                                "account": "บัญชีผู้ใช้ รหัสผ่าน สมัคร/ลบบัญชี", "sales": "สั่งซื้อ เปลี่ยนแพ็กเกจ สมัครบริการ",
                                "support": "ปัญหาการใช้งาน ร้องเรียน ขอคุยกับพนักงาน เรื่องทั่วไป"}},
    "urgency": {"type": "score", "instructions": "เรื่องนี้เร่งด่วนแค่ไหน",
                "criteria": ["ไม่รีบ ถามข้อมูลทั่วไป", "ควรตอบภายในวันนี้", "ด่วน ลูกค้าเสียหายหรือใช้งานไม่ได้อยู่"]},
    "frustration": {"type": "score", "instructions": "ลูกค้าไม่พอใจแค่ไหน", "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]},
    "wants_refund": {"type": "noul", "instructions": "ลูกค้าขอเงินคืนหรือถามเรื่องการคืนเงินหรือไม่"},
    "wants_human": {"type": "noul", "instructions": "ลูกค้าต้องการคุยกับพนักงานที่เป็นคนจริงหรือไม่"},
    "has_order_ref": {"type": "noul", "instructions": "ข้อความระบุหมายเลขคำสั่งซื้อ ใบแจ้งหนี้ หรือเลขอ้างอิงหรือไม่"},
    "sentiment": {"type": "choice", "instructions": "อารมณ์โดยรวมของข้อความ",
                  "criteria": {"positive": "ชม พอใจ ดีใจ", "neutral": "เล่าเฉย ๆ ให้ข้อมูล", "negative": "บ่น ไม่พอใจ โกรธ", "question": "ถามคำถาม ขอข้อมูล"}},
}
WISESIGHT = {0: "positive", 1: "neutral", 2: "negative", 3: "question"}


def one_hot(q, label):
    keys = list(q["criteria"])
    v = [0.0] * len(keys)
    v[keys.index(label)] = 1.0
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/thai/data/cc")
    ap.add_argument("--teacher", default="http://localhost:8010", help="comma-separated teacher URLs, used round-robin")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--wisesight", type=int, default=12000)
    ap.add_argument("--eval-frac", type=float, default=0.05)
    ap.add_argument("--questions-per-record", type=int, default=2, help="items per record: the human-labelled question + sampled teacher questions")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run3", help="tokenizer + config (head_max_len) to tokenise with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="smoke: only this many records per source")
    ap.add_argument("--from-records", action="store_true", help="skip the teacher: rebuild the split and items from cc.jsonl + cc_eval.jsonl already on disk")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    data = Path(args.data)
    teachers = args.teacher.split(",")

    # ---- records
    recs = []
    for i, line in enumerate(open(data / "porameht_colloquial_kept.jsonl", encoding="utf-8")):
        r = json.loads(line)
        recs.append({"id": f"cc-{i}", "source": "cc", "group": r["src"], "state": r["text"], "style": r["style"],
                     "labels": {"intent": r["intent"], "category": r["category"]}})
    ws = [json.loads(l) for l in open(data / "wisesight_train.jsonl", encoding="utf-8")]
    ws = [r for r in ws if len(r["text"].strip()) >= 10]
    rng.shuffle(ws)
    for i, r in enumerate(ws[: args.wisesight]):
        recs.append({"id": f"ws-{i}", "source": "wisesight", "group": f"ws-{i}", "state": r["text"].strip(),
                     "labels": {"sentiment": WISESIGHT[int(r["label"])]}})
    if args.limit:
        recs = [r for r in recs if r["source"] == "cc"][: args.limit] + [r for r in recs if r["source"] == "wisesight"][: args.limit]
    for r in recs:
        r["questions"] = QUESTIONS
    print(f"{len(recs)} records ({Counter(r['source'] for r in recs)})", flush=True)

    # ---- teacher (or reuse labelled records)
    t0 = time.perf_counter()
    done = [0]
    if args.from_records:
        labelled = [json.loads(l) for f in ("cc.jsonl", "cc_eval.jsonl") for l in open(data / f, encoding="utf-8")]
        for r in labelled:
            r["questions"] = QUESTIONS
        print(f"reusing {len(labelled)} labelled records from disk", flush=True)

    def work(ir):
        i, r = ir
        res = ask_teacher(teachers[i % len(teachers)], r["state"], r["questions"])
        done[0] += 1
        if done[0] % 1000 == 0:
            el = time.perf_counter() - t0
            print(f"  {done[0]}/{len(recs)} labelled, {el / 60:.1f} min, {done[0] / el:.1f} rec/s", flush=True)
        if res is None:
            return None
        targets = targets_from(res["answers"], r["questions"])
        for qid, lab in r["labels"].items():
            targets[qid] = one_hot(r["questions"][qid], lab)
        return {**r, "targets": targets, "teacher_tokens": res["usage"]["input_tokens"]}

    if not args.from_records:
        with ThreadPoolExecutor(args.workers) as ex:
            labelled = [x for x in ex.map(work, enumerate(recs)) if x]
        print(f"teacher labelled {len(labelled)}/{len(recs)} in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    # ---- split by source sentence
    groups = sorted({r["group"] for r in labelled})
    rng.shuffle(groups)
    eval_groups = set(groups[: int(len(groups) * args.eval_frac)])
    ev = [r for r in labelled if r["group"] in eval_groups]
    tr = [r for r in labelled if r["group"] not in eval_groups]
    rng.shuffle(tr)
    for name, rows in (("cc.jsonl", tr), ("cc_eval.jsonl", ev)):  # eval keeps `questions`: eval_thai.py --teacher reads it
        with open(data / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k != "questions" or name == "cc_eval.jsonl"}, ensure_ascii=False) + "\n")
    with open(data / "cc_eval_human.jsonl", "w", encoding="utf-8") as f:  # eval_thai.py format
        for r in ev:
            f.write(json.dumps({"id": r["id"], "source": r["source"], "state": r["state"],
                                "questions": {q: QUESTIONS[q] for q in r["labels"]}, "labels": r["labels"]}, ensure_ascii=False) + "\n")

    # ---- items
    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))
    items, dropped, by_type = [], 0, Counter()
    for r in tr:
        first = rng.choice(list(r["labels"]))  # one human-labelled question (intent or category for cc rows) ...
        others = [q for q in QUESTIONS if q != first]  # ... plus sampled others, the second labelled one included
        chosen = [first] + rng.sample(others, max(0, args.questions_per_record - 1))
        for qid in chosen:
            q = QUESTIONS[qid]
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
    torch.save(items, data / "cc_items.pt")
    summary = {"records": len(labelled), "train_records": len(tr), "eval_records": len(ev), "items": len(items), "dropped": dropped,
               "by_type": dict(by_type), "questions_per_record": args.questions_per_record,
               "head_max_len": cfg["head_max_len"], "max_len": cfg["max_len"],
               "mean_len": sum(len(i["ids"]) for i in items) / max(1, len(items)), "teachers": teachers,
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (data / "cc_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
