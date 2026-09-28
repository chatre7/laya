"""Second hand-labelling batch: sample more Pantip questions per business, excluding every topic already in the review sample
(the eval set) and in the first extra batch (run 8 training). Writes data_domain/pantip_<biz>_extra2.jsonl + _print_<biz>_extra2_<k>.txt"""
import json
import os
import random

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_domain")
N = {"telecom": 500, "banking": 500, "insurance": 200}
random.seed(29)
for biz, n in N.items():
    used = set()
    for name in (f"pantip_{biz}_review_sample.jsonl", f"pantip_{biz}_extra.jsonl"):
        used |= {json.loads(l)["topic_id"] for l in open(os.path.join(DATA, name), encoding="utf-8")}
    rows = [json.loads(l) for l in open(os.path.join(DATA, f"pantip_{biz}.jsonl"), encoding="utf-8")]
    pool = [r for r in rows if r["topic_id"] not in used and r["question_like"] and 20 <= len(r["text"]) <= 400]
    s = random.sample(pool, min(n, len(pool)))
    with open(os.path.join(DATA, f"pantip_{biz}_extra2.jsonl"), "w", encoding="utf-8") as f:
        for r in s:
            f.write(json.dumps({"text": r["text"], "topic_id": r["topic_id"]}, ensure_ascii=False) + "\n")
    for k in range(0, len(s), 100):
        with open(os.path.join(DATA, f"_print_{biz}_extra2_{k // 100}.txt"), "w", encoding="utf-8") as f:
            for i, r in enumerate(s[k:k + 100], start=k):
                f.write(f"{i:3d}| " + r["text"][:300].replace("\n", " ") + "\n")
    print(biz, "pool", len(pool), "sampled", len(s), "chunks", (len(s) + 99) // 100)
