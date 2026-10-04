"""Load one Decision 2.0 model (vllm-sr, Hugging Face) and answer our call-center questions on a few Thai messages:
what do the answers look like, how long does a request take on this GPU, how much memory does it hold.

    python smoke.py vllm-sr/Decision-2.0-Kai-0.6B
"""
import json
import os
import sys
import time

import torch
from transformers import AutoModel

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cs"))
from cs_questions import SHARED, intent_question  # noqa: E402

TEXTS = [
    ("banking", "โดนหักเงินซ้ำสองครั้งเมื่อวานนี้ ขอเงินคืนด่วนนะครับ โทรไปสามรอบแล้วไม่มีใครรับ"),
    ("telecom", "เน็ตบ้านหลุดบ่อยมากตั้งแต่เมื่อคืน รีสตาร์ทเราเตอร์แล้วก็ยังไม่ได้ ช่วยดูให้หน่อยครับ"),
    ("banking", "ระบบทำงานได้สมบูรณ์แบบมากจ้า ค้างไปแค่ 10 รอบเอง"),
    ("insurance", "สอบถามครับ ประกันรถชั้น 1 เคลมกระจกได้ไหม"),
]

repo = sys.argv[1]
model = AutoModel.from_pretrained(repo, trust_remote_code=True)
print("loaded", repo, "| gpu MiB", torch.cuda.memory_allocated() // 2**20, flush=True)
for biz, text in TEXTS:
    qs = {"intent": intent_question(biz), **{k: SHARED[k] for k in ("department", "urgency", "frustration", "wants_refund", "sentiment")}}
    out = model.system_one(state=text, questions=qs)
    t = time.perf_counter()
    out = model.system_one(state=text, questions=qs)
    ms = (time.perf_counter() - t) * 1000
    print(f"\n{ms:.0f} ms, {out['usage']['input_tokens']} tokens | {text[:60]}")
    for qid, a in out["answers"].items():
        print("  ", qid, json.dumps(a, ensure_ascii=False)[:300])
print("\npeak gpu MiB", torch.cuda.max_memory_allocated() // 2**20)
