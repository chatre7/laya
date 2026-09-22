"""Build a distillation set for laya from OpenThai-SystemOne (the teacher).

Any Thai text + a bank of typed questions -> the teacher's probabilities become soft targets.
laya already trains on soft targets, so no human labels are needed and the data covers whatever
questions we care about instead of whatever public datasets happen to label.

    python distill_from_ots.py --out /work/thai/data --per-source 3000 --teacher http://172.18.72.145:8010

Writes  distill.jsonl        records {id, source, state, questions, targets}
        distill_eval.jsonl   5% held out (student vs teacher agreement)
        distill_items.pt     tokenised training items with soft targets (train_single.py format)
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_sequence, render_options

# ----------------------------------------------------------------------------- Thai text sources
def load(name, config=None, split="train", **kw):
    from datasets import load_dataset
    return load_dataset(name, config, split=split, streaming=True, **kw)


def take(it, n):
    for i, x in enumerate(it):
        if i >= n:
            break
        yield x


def texts(source, n):
    """Yield (kind, state) pairs. kind steers which questions make sense."""
    if source == "prachathai":  # news; eval uses the validation split, so train only
        for ex in take(load("PyThaiNLP/prachathai67k", split="train", revision="refs/convert/parquet"), n):
            yield "news", {"title": ex["title"], "body": ex["body_text"][:1500]}
    elif source == "wisesight":  # social posts; test split is the human eval set, so train only
        for ex in take(load("pythainlp/wisesight_sentiment", split="train"), n):
            yield "social", ex["texts"]
    elif source == "wongnai":
        for ex in take(load("Wongnai/wongnai_reviews", split="train"), n):
            yield "review", ex["review_body"][:1500]
    elif source == "massive":
        for ex in take(load("mteb/amazon_massive_intent", "th", split="train"), n):
            yield "utterance", ex["text"]
    elif source == "toxicity":
        for ex in take(load("tmu-nlp/thai_toxicity_tweet", split="train", revision="refs/convert/parquet"), n):
            if ex.get("tweet_text") and ex["tweet_text"] != "TWEET_NOT_FOUND":
                yield "social", ex["tweet_text"]
    elif source == "xnli":
        for ex in take(load("facebook/xnli", "th", split="train"), n):
            yield "pair", {"premise": ex["premise"], "hypothesis": ex["hypothesis"]}
    else:
        raise ValueError(source)


# ----------------------------------------------------------------------------- question bank
SENT3 = {"เชิงบวก": "พอใจ ชื่นชม ยินดี", "เป็นกลาง": "บอกเล่า ถามข้อมูล ไม่แสดงอารมณ์", "เชิงลบ": "ไม่พอใจ ตำหนิ โกรธ"}
SENT4 = {**SENT3, "คำถาม": "ถามข้อมูลหรือขอความช่วยเหลือ"}
TOPICS = ["การเมือง", "เศรษฐกิจ", "สังคม", "กีฬา", "บันเทิง", "เทคโนโลยี", "สุขภาพ", "การศึกษา", "สิ่งแวดล้อม", "ต่างประเทศ", "อาชญากรรม", "ท่องเที่ยว"]
DEPTS = {"billing": "ค่าบริการ ใบแจ้งหนี้ การชำระเงิน คืนเงิน", "technical": "ระบบ แอป สัญญาณ ใช้งานไม่ได้", "sales": "สมัคร เปลี่ยนแพ็กเกจ โปรโมชั่น ราคา",
         "other": "เรื่องอื่นที่ไม่เข้าสามหมวดข้างต้น"}
INTENTS = ["ถามข้อมูล", "ร้องเรียน", "ขอความช่วยเหลือ", "ชื่นชม", "สั่งซื้อหรือจอง", "ยกเลิก", "ขอเงินคืน", "แจ้งปัญหา", "ทักทาย", "อื่น ๆ"]
FOOD_ASPECTS = ["รสชาติ", "ราคา", "บริการ", "บรรยากาศ", "ความสะอาด", "ที่จอดรถ", "การรอคิว"]


def question_bank(kind, rng):
    """A few questions that make sense for this kind of text, with paraphrased instructions."""
    qs = {}
    if kind in ("social", "review", "utterance"):
        qs["sentiment"] = {"type": "choice", "instructions": rng.choice(["ข้อความนี้แสดงความรู้สึกแบบใด", "อารมณ์ของผู้เขียนเป็นอย่างไร", "จัดประเภทความรู้สึกของข้อความ"]),
                           "criteria": rng.choice([SENT3, SENT4])}
    if kind == "review":
        qs["rating"] = {"type": "score", "instructions": rng.choice(["รีวิวนี้ให้กี่ดาว", "ประเมินความพึงพอใจของผู้รีวิว"]),
                        "criteria": ["แย่มาก", "แย่", "ปานกลาง", "ดี", "ดีมาก"]}
        asp = rng.sample(FOOD_ASPECTS, 3)
        qs["aspect"] = {"type": "choice", "instructions": "รีวิวนี้พูดถึงเรื่องใดเป็นหลัก", "criteria": {a: None for a in asp + ["ไม่มีข้อใดตรง"]}}
        qs["recommend"] = {"type": "noul", "instructions": rng.choice(["ผู้รีวิวแนะนำร้านนี้หรือไม่", "ผู้เขียนน่าจะกลับมาใช้บริการอีกหรือไม่"])}
    if kind in ("social", "utterance"):
        qs["intent"] = {"type": "choice", "instructions": rng.choice(["ผู้เขียนต้องการทำอะไร", "เจตนาของข้อความนี้คืออะไร"]),
                        "criteria": {i: None for i in rng.sample(INTENTS, rng.choice([4, 6, 10]))}}
        qs["dept"] = {"type": "choice", "instructions": rng.choice(["ถ้าเป็นข้อความถึงฝ่ายบริการลูกค้า ทีมใดควรรับผิดชอบ", "ทีมใดควรตอบข้อความนี้"]), "criteria": DEPTS}
        qs["urgency"] = {"type": "score", "instructions": rng.choice(["ข้อความนี้เร่งด่วนแค่ไหน", "ควรตอบเร็วแค่ไหน"]), "criteria": ["ไม่เร่งด่วน", "ปานกลาง", "ด่วนมาก"]}
        qs["frustration"] = {"type": "score", "instructions": rng.choice(["ผู้เขียนไม่พอใจแค่ไหน", "ระดับความหงุดหงิดของผู้เขียน"]), "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]}
        qs["is_question"] = {"type": "noul", "instructions": rng.choice(["ข้อความนี้เป็นคำถามหรือไม่", "ผู้เขียนกำลังถามอะไรบางอย่างใช่หรือไม่"])}
        qs["toxic"] = {"type": "noul", "instructions": "ข้อความนี้มีการด่าทอ เหยียดหยาม หรือคุกคามหรือไม่"}
        qs["refund"] = {"type": "noul", "instructions": "ผู้เขียนขอเงินคืนอย่างชัดเจนหรือไม่"}
    if kind == "news":
        qs["topic"] = {"type": "choice", "instructions": rng.choice(["ข่าวนี้อยู่ในหมวดใด", "หัวข้อหลักของข่าวนี้คืออะไร"]),
                       "criteria": {t: None for t in rng.sample(TOPICS, rng.choice([5, 8, 12]))}}
        t = rng.choice(TOPICS)
        qs["about"] = {"type": "noul", "instructions": f"ข่าวนี้เกี่ยวข้องกับ '{t}' หรือไม่"}
        qs["tone"] = {"type": "score", "instructions": "น้ำเสียงของข่าวเป็นอย่างไร", "criteria": ["เชิงลบ", "เป็นกลาง", "เชิงบวก"]}
        qs["thailand"] = {"type": "noul", "instructions": "ข่าวนี้เกิดขึ้นในประเทศไทยหรือไม่"}
    if kind == "pair":
        qs["nli"] = {"type": "choice", "instructions": rng.choice(["ความสัมพันธ์ระหว่าง premise และ hypothesis คืออะไร", "hypothesis สอดคล้องกับ premise แค่ไหน"]),
                     "criteria": {"entailment": "ประโยคที่สองสรุปได้จากประโยคแรก", "neutral": "ไม่สามารถสรุปได้", "contradiction": "ประโยคที่สองขัดแย้งกับประโยคแรก"}}
        qs["entails"] = {"type": "noul", "instructions": "hypothesis สรุปได้จาก premise หรือไม่"}
        qs["same_topic"] = {"type": "noul", "instructions": "สองประโยคพูดถึงเรื่องเดียวกันหรือไม่"}
    keys = list(qs)
    rng.shuffle(keys)
    return {k: qs[k] for k in keys[: rng.choice([2, 3, 4])]}


# ----------------------------------------------------------------------------- teacher
def ask_teacher(url, state, questions):
    body = json.dumps({"state": state, "questions": questions, "order_invariant": False}, ensure_ascii=False).encode()
    req = urllib.request.Request(f"{url}/v1/systemone", body, {"content-type": "application/json"})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))
        except Exception as e:  # 503 under load, transient network: back off and retry
            if attempt == 3:
                print("teacher failed:", type(e).__name__, str(e)[:100], file=sys.stderr)
                return None
            time.sleep(2 * (attempt + 1))


def targets_from(answers, questions):
    """Teacher answer -> probability vector in laya option order (abstain mass dropped, renormalised)."""
    out = {}
    for qid, q in questions.items():
        a = answers[qid]
        if q["type"] == "choice":
            v = [float(a["probabilities"][k]) for k in q["criteria"]]
        elif q["type"] == "score":
            v = [float(a["probabilities"][str(i)]) for i in range(len(q["criteria"]))]
        else:
            v = [1.0 - float(a["noul"]), float(a["noul"])]
        s = sum(v)
        out[qid] = [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/thai/data")
    ap.add_argument("--teacher", default="http://172.18.72.145:8010")
    ap.add_argument("--sources", default="prachathai,wisesight,wongnai,massive,toxicity,xnli")
    ap.add_argument("--per-source", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default="convaiinnovations/laya-multilingual", help="student tokenizer/config")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    jobs = []
    for src in args.sources.split(","):
        n = 0
        try:
            for kind, state in texts(src, args.per_source):
                jobs.append({"id": f"{src}-{n}", "source": src, "kind": kind, "state": state, "questions": question_bank(kind, rng)})
                n += 1
        except Exception as e:
            print(f"{src}: FAILED {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)
        print(f"{src}: {n} texts", flush=True)

    t0 = time.perf_counter()
    done = 0

    def work(j):
        r = ask_teacher(args.teacher, j["state"], j["questions"])
        return None if r is None else {**j, "targets": targets_from(r["answers"], j["questions"]), "teacher_tokens": r["usage"]["input_tokens"]}

    records = []
    with ThreadPoolExecutor(args.workers) as ex:
        for rec in ex.map(work, jobs):
            done += 1
            if rec:
                records.append(rec)
            if done % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  {done}/{len(jobs)} labelled, {el / 60:.1f} min, {done / el:.1f} rec/s", flush=True)
    print(f"teacher labelled {len(records)}/{len(jobs)} records in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)

    rng.shuffle(records)
    n_eval = max(200, len(records) // 20)
    with open(out / "distill_eval.jsonl", "w", encoding="utf-8") as f:
        for r in records[:n_eval]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "distill.jsonl", "w", encoding="utf-8") as f:
        for r in records[n_eval:]:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    model_dir = snapshot_download(args.model, allow_patterns=["rl_agent_config.json", "tokenizer/*", "encoder/*"])
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    items, dropped, by_type = [], 0, [0, 0, 0]
    for r in records[n_eval:]:
        for qid, q in r["questions"].items():
            crit = q.get("criteria") if q["type"] != "noul" else (q.get("criteria") or {})
            qi = {"t": q["type"], "ins": q["instructions"], "crit": crit}
            seq, markers = build_sequence(tok, r["state"], qi, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(qi)):
                dropped += 1
                continue
            target = r["targets"][qid]
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]], "target": target,
                          "label": max(range(len(target)), key=target.__getitem__), "source": r["source"]})
            by_type[QTYPES[q["type"]]] += 1
    rng.shuffle(items)
    torch.save(items, out / "distill_items.pt")
    summary = {"records": len(records), "eval_records": n_eval, "items": len(items), "dropped": dropped,
               "by_type": {"choice": by_type[0], "score": by_type[1], "noul": by_type[2]},
               "mean_len": sum(len(i["ids"]) for i in items) / max(1, len(items)),
               "minutes": round((time.perf_counter() - t0) / 60, 1)}
    (out / "distill_manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
