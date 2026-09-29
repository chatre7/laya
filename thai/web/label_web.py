"""Web-agent domain items for laya: Mind2Web element choice (target) + operation, in the demo/browser_agent.py request format
(state = {"task", "previous", "screen"}), task in English and in Thai, question wording in English and in Thai. Labels are the
dataset's human actions (one-hot, smoothing 0.1); no teacher. Also writes eval files for the three Mind2Web test splits.

    python label_web.py --data /work/thai/data/web --student /work/thai/out/laya-th-run3
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402

TARGET_INS = ["Which element should the agent act on next?", "ควรทำงานกับ element ไหนต่อ", "ขั้นต่อไปควรคลิกหรือกรอกที่ไหน"]
OP_INS = ["Which operation should be performed on it?", "ควรทำอะไรกับ element นั้น"]
OPS = ["CLICK", "TYPE", "SELECT"]


def questions(rec, rng, lang_ins=None):
    ti = TARGET_INS[0] if lang_ins == "en" else (TARGET_INS[1] if lang_ins == "th" else rng.choice(TARGET_INS))
    oi = OP_INS[0] if lang_ins == "en" else (OP_INS[1] if lang_ins == "th" else rng.choice(OP_INS))
    return {"target": {"type": "choice", "instructions": ti, "criteria": {o: None for o in rec["options"]}},
            "operation": {"type": "choice", "instructions": oi, "criteria": {o: None for o in OPS}}}


def state_of(rec, task):
    st = {"task": task}
    if rec["previous"]:
        st["previous"] = " ; ".join(rec["previous"])
    st["screen"] = "\n".join(rec["options"])
    return st


def one_hot(keys, label, smooth):
    v = [smooth / len(keys)] * len(keys)
    v[keys.index(label)] += 1.0 - smooth
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/work/thai/data/web")
    ap.add_argument("--student", default="/work/thai/out/laya-th-run3")
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    data = Path(args.data)
    th = json.load(open(data / "tasks_th.json", encoding="utf-8")) if (data / "tasks_th.json").exists() else {}
    print(f"{len(th)} Thai task translations")
    _fix_tokenizer_config(args.student)
    tok = AutoTokenizer.from_pretrained(os.path.join(args.student, "tokenizer"))
    cfg = json.load(open(os.path.join(args.student, "rl_agent_config.json")))

    # ---- train items
    recs = [json.loads(l) for l in open(data / "m2w_train.jsonl", encoding="utf-8")]
    items, dropped, by = [], 0, Counter()
    for r in recs:
        tasks = [("en", r["task_en"])] + ([("th", th[r["task_en"]])] if r["task_en"] in th else [])
        for lang, task in tasks:
            qs = questions(r, rng)
            st = state_of(r, task)
            for qid, label in (("target", r["target"]), ("operation", r["op"])):
                q = qs[qid]
                qi = {"t": "choice", "ins": q["instructions"], "crit": q["criteria"]}
                seq, markers = build_sequence(tok, st, qi, cfg["max_len"], cfg["head_max_len"])
                if len(markers) != len(render_options(qi)):
                    dropped += 1
                    continue
                target = one_hot(list(q["criteria"]), label, args.smooth)
                items.append({"ids": seq, "markers": markers, "qtype": QTYPES["choice"], "target": target,
                              "label": max(range(len(target)), key=target.__getitem__), "source": f"m2w_{lang}_{qid}"})
                by[f"{lang}_{qid}"] += 1
    rng.shuffle(items)
    torch.save(items, data / "web1_items.pt")
    print(f"train: {len(recs)} records -> {len(items)} items, dropped {dropped} (options did not fit), {dict(by)}")

    # ---- eval files (eval_thai.py format): Thai task + Thai questions, and English task + English questions
    for split in ("test_task", "test_website", "test_domain"):
        p = data / f"m2w_{split}.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        for lang in ("th", "en"):
            n = 0
            with open(data / f"web_eval_{split}_{lang}.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    task = th.get(r["task_en"]) if lang == "th" else r["task_en"]
                    if not task:
                        continue
                    f.write(json.dumps({"id": r["id"], "source": f"{split}_{lang}", "state": state_of(r, task), "questions": questions(r, rng, lang),
                                        "labels": {"target": r["target"], "operation": r["op"]}}, ensure_ascii=False) + "\n")
                    n += 1
            print(f"eval {split} {lang}: {n}")


if __name__ == "__main__":
    main()
