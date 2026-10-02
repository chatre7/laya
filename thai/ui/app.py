"""Test desk for the call-center team: type a customer message, see what the served model (:8011) decides (merged intent
group, department, urgency, frustration, next best action), and mark it right or wrong. Every right/wrong click appends one
line to FEEDBACK_FILE: that is human-labelled real text, the thing this project lacks most. Nothing is stored on analyze.

    CASCADE_URL=http://172.18.72.145:8011 FEEDBACK_FILE=/data/feedback.jsonl uvicorn app:app --host 0.0.0.0 --port 8020
"""
import json
import os
import sys
import threading
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cs"))
from cs_questions import INTENTS, SHARED, intent_question  # noqa: E402
from intent_groups import GROUPS, group_probs  # noqa: E402
from nba_actions import PLAYBOOKS, apply_context  # noqa: E402

CASCADE = os.environ.get("CASCADE_URL", "http://172.18.72.145:8011")
FEEDBACK = Path(os.environ.get("FEEDBACK_FILE", "/data/feedback.jsonl"))
HTML = Path(__file__).with_name("index.html")
LOCK = threading.Lock()
BANGKOK = timezone(timedelta(hours=7))  # the container clock is UTC and the image has no tzdata

BUSINESS_TH = {"banking": "ธนาคาร", "telecom": "ค่ายมือถือ / อินเทอร์เน็ต", "insurance": "ประกัน"}
DEPT_TH = {"billing": "การเงิน / ค่าบริการ", "technical": "เทคนิค", "account": "บัญชี / ข้อมูลลูกค้า", "sales": "ขาย",
           "claims": "สินไหม", "support": "บริการทั่วไป"}
OTHER_TH = "ไม่เกี่ยวข้อง หรือไม่เข้าหมวดใด"
URGENCY_TH = ["ไม่รีบ", "ควรตอบภายในวันนี้", "ด่วน"]
FRUSTRATION_TH = ["ใจเย็น", "หงุดหงิด", "โกรธมาก"]
SAMPLES = {
    "banking": ["โอนเงินไปตั้งแต่เมื่อวาน ยังไม่เข้าเลยครับ", "ทำบัตรเครดิตหาย ต้องทำยังไงคะ", "แอปเข้าไม่ได้ ขึ้นว่ารหัสผิด ลองมาทั้งวันแล้ว",
                "อยากสมัครบัตรเครดิต เงินเดือน 18,000 ได้ไหมคะ", "โดนหักเงินซ้ำสองครั้ง ขอเงินคืนด้วยครับ"],
    "telecom": ["เน็ตบ้านหลุดทุกคืน รีสตาร์ทแล้วก็ไม่หาย", "เติมเงินผิดเบอร์ 200 บาท ขอคืนได้ไหม", "อยากเปลี่ยนโปรเป็นเน็ตไม่อั้น ราคาไม่เกิน 300",
                "บิลเดือนนี้แพงกว่าปกติ มีค่าอะไรไม่รู้เพิ่มมา", "จะย้ายค่ายแล้ว บริการแย่มาก"],
    "insurance": ["รถชนเมื่อกี้ที่แยก ต้องแจ้งเคลมยังไงครับ", "ยื่นเคลมไปสองอาทิตย์แล้ว ยังไม่ได้เงินเลย", "สนใจประกันสุขภาพให้ลูก 2 ขวบ มีแผนไหนบ้าง",
                  "จ่ายเบี้ยไปแล้วแต่ระบบยังขึ้นว่าค้างชำระ", "ยังไม่ได้รับกรมธรรม์เลยค่ะ ซื้อไปเดือนกว่าแล้ว"],
}

app = FastAPI(title="laya test desk", docs_url=None, redoc_url=None)


class AnalyzeIn(BaseModel):
    business: Literal["banking", "telecom", "insurance"]
    message: str = Field(min_length=2, max_length=2000)
    verified: bool = True
    contacts: Literal[1, 3] = 1


class FeedbackIn(AnalyzeIn):
    correct: bool
    predicted_group: str
    predicted_department: str
    right_group: Optional[str] = None
    right_department: Optional[str] = None
    note: str = Field(default="", max_length=500)
    tester: str = Field(default="", max_length=60)


def level(score: float) -> int:
    return max(0, min(2, round(score)))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML.read_text(encoding="utf-8")


@app.get("/api/meta")
def meta():
    return {"businesses": [{"id": b, "name": BUSINESS_TH[b], "samples": SAMPLES[b],
                            "groups": [{"id": g, "name": d} for g, (d, _) in GROUPS[b].items()] + [{"id": "other", "name": OTHER_TH}]}
                           for b in ("banking", "telecom", "insurance")],
            "departments": [{"id": k, "name": v} for k, v in DEPT_TH.items()]}


@app.post("/api/analyze")
def analyze(inp: AnalyzeIn):
    biz = inp.business
    questions = {"intent": intent_question(biz), "department": SHARED["department"], "urgency": SHARED["urgency"], "frustration": SHARED["frustration"]}
    body = json.dumps({"state": inp.message.strip(), "questions": questions}, ensure_ascii=False).encode()
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{CASCADE}/v1/systemone", body, {"content-type": "application/json"}), timeout=60) as r:
            ans = json.load(r)["answers"]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"ระบบวิเคราะห์ (:8011) ไม่ตอบ: {type(e).__name__}") from e
    ms = round((time.perf_counter() - t) * 1000)
    probs = ans["intent"]["probabilities"]
    gp = sorted(group_probs(biz, probs).items(), key=lambda kv: -kv[1])[:3]
    names = {g: d for g, (d, _) in GROUPS[biz].items()}
    names["other"] = OTHER_TH
    fine = max(probs, key=probs.get)
    pb = PLAYBOOKS[biz]
    score = defaultdict(float)
    for intent, p in probs.items():
        score[apply_context(biz, pb["intent_default"].get(intent, "out_of_scope"), inp.verified, inp.contacts)] += p
    actions = sorted(score.items(), key=lambda kv: -kv[1])[:3]
    dept = ans["department"]["choice"]
    return {"groups": [{"id": g, "name": names[g], "p": round(p, 3)} for g, p in gp],
            "fine": {"id": fine, "name": INTENTS[biz].get(fine, OTHER_TH), "p": round(probs[fine], 3)},
            "department": {"id": dept, "name": DEPT_TH.get(dept, dept), "p": round(ans["department"]["probabilities"][dept], 3)},
            "urgency": {"level": level(ans["urgency"]["score"]), "label": URGENCY_TH[level(ans["urgency"]["score"])], "score": round(ans["urgency"]["score"], 2)},
            "frustration": {"level": level(ans["frustration"]["score"]), "label": FRUSTRATION_TH[level(ans["frustration"]["score"])],
                            "score": round(ans["frustration"]["score"], 2)},
            "actions": [{"id": a, "name": pb["actions"][a], "p": round(p, 3)} for a, p in actions], "ms": ms}


@app.post("/api/feedback")
def feedback(fb: FeedbackIn):
    rec = {"at": datetime.now(BANGKOK).isoformat(timespec="seconds"), **fb.model_dump()}
    with LOCK:
        FEEDBACK.parent.mkdir(parents=True, exist_ok=True)
        with open(FEEDBACK, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"saved": True, **stats()}


@app.get("/api/stats")
def stats():
    n = ok = 0
    if FEEDBACK.exists():
        for line in open(FEEDBACK, encoding="utf-8"):
            if line.strip():
                n += 1
                ok += bool(json.loads(line).get("correct"))
    return {"total": n, "correct": ok}


@app.get("/healthz")
def healthz():
    return {"ok": True, "cascade": CASCADE}
