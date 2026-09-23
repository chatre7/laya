"""Student -> teacher cascade server: laya-thai-distill-run3 answers first, OpenThai-SystemOne answers what it is unsure about.

Same request/response contract as the OpenThai-SystemOne batching server (docker/server_batched.py in
chatre7/system-one), so a client switches between :8010 (teacher only) and :8011 (cascade) by URL alone.

Per question: the student answers in one forward pass over all questions; a question goes to the teacher when
the student's confidence (max probability, the quantity cascade.py measured) is below CASCADE_THRESHOLD, when it
has more than CASCADE_MAX_OPTIONS options (0 = no limit), or when the student cannot encode it (options exceed its
768-token budget). All such questions are sent to the teacher in one follow-up request with the same state and the
answers are merged. `usage.cascade` reports which questions reached the teacher and why.

Measured on 2,455 human-labelled decisions (thai/results/cascade3_opt60.json): threshold 0.7 without an option gate
gives accuracy 0.803 vs 0.816 teacher-only, with 28% of questions reaching the teacher; ~280 ms per record under
8 concurrent callers vs ~910 ms teacher-only.

    LAYA_STUDENT=/model TEACHER_URL=http://172.18.72.145:8010 uvicorn cascade_server:app --host 0.0.0.0 --port 8011
"""
import asyncio
import json
import os
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request

STUDENT = os.environ.get("LAYA_STUDENT", "/work/thai/out/laya-th-run3")
TEACHER_URL = os.environ.get("TEACHER_URL", "http://172.18.72.145:8010").rstrip("/")
THRESHOLD = float(os.environ.get("CASCADE_THRESHOLD", "0.7"))
MAX_OPTIONS = int(os.environ.get("CASCADE_MAX_OPTIONS", "0"))
TEACHER_TIMEOUT = float(os.environ.get("TEACHER_TIMEOUT_S", "120"))
MODEL_NAME = os.environ.get("CASCADE_MODEL_NAME", "laya-thai-distill-run3+openthai-systemone")
QTYPES = ("choice", "score", "noul")


# ---------------------------------------------------------------- routing policy
def student_confidence(q: Dict[str, Any], a: Dict[str, Any]) -> float:
    """The gate quantity: max probability (noul: max(p, 1-p)). This is what cascade.py swept, not laya's
    entropy-based `confidence` field, so the measured curve applies to this server as is."""
    if q["type"] == "noul":
        return max(float(a["noul"]), 1.0 - float(a["noul"]))
    return max(float(v) for v in a["probabilities"].values())


def n_options(q: Dict[str, Any]) -> int:
    if q["type"] == "noul":
        return 2
    c = q.get("criteria")
    return len(c) if c else 0


def route(q: Dict[str, Any], a: Dict[str, Any], conf: float) -> Optional[str]:
    """None = keep the student's answer; otherwise the reason this question goes to the teacher.
    One global threshold for every question type, as measured. Per-type thresholds, an abstain proxy
    (low max-prob AND flat distribution) or a per-request override would go here."""
    if MAX_OPTIONS and n_options(q) > MAX_OPTIONS:
        return "options>%d" % MAX_OPTIONS
    if conf < THRESHOLD:
        return "confidence<%.2f" % THRESHOLD
    return None


# ---------------------------------------------------------------- models
class Student:
    """laya agent behind a lock: one forward at a time on the GPU, called from a worker thread."""

    def __init__(self, path: str):
        import laya  # imported here so the module loads without torch (tests, docs)

        self.agent = laya.Agent(path, device="cuda")
        self.cfg = dict(self.agent.cfg)
        self.lock = threading.Lock()

    def predict(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        with self.lock:
            out = self.agent.system_one(state, questions)
        for qid, a in out["answers"].items():
            if a.get("type") == "score" and "legend" not in a:  # match the teacher's ScoreAnswer shape
                a["legend"] = {i: c for i, c in enumerate(questions[qid].get("criteria") or [])}
        return out


def teacher_call(body: Dict[str, Any]) -> Dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(TEACHER_URL + "/v1/systemone", data, {"content-type": "application/json"})
    last: Optional[BaseException] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=TEACHER_TIMEOUT) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 422:  # the teacher rejected the request shape: not retryable, surface it
                raise HTTPException(status_code=422, detail="teacher: " + e.read().decode("utf-8", "replace"))
            last = e  # 503 busy etc.
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(0.5 * (attempt + 1))
    raise RuntimeError("teacher unavailable after 3 attempts: %r" % (last,))


def teacher_healthy() -> bool:
    try:
        with urllib.request.urlopen(TEACHER_URL + "/healthz", timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------- app
@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    app.state.student = Student(STUDENT)
    app.state.student.predict("ทดสอบระบบ", {"w": {"type": "noul", "instructions": "เป็นข้อความทดสอบหรือไม่"}})  # warm-up
    app.state.stats = {"requests": 0, "questions": 0, "teacher_calls": 0, "teacher_questions": 0,
                       "student_errors": 0, "teacher_failures": 0, "student_ms_total": 0.0, "teacher_ms_total": 0.0}
    print("[cascade] student %s loaded in %.1f s, head_max_len=%s, teacher %s, threshold %.2f, max_options %d"
          % (STUDENT, time.perf_counter() - t, app.state.student.cfg.get("head_max_len"), TEACHER_URL, THRESHOLD, MAX_OPTIONS), flush=True)
    yield


app = FastAPI(
    title="laya -> OpenThai-SystemOne cascade",
    version="0.1.0",
    description=(
        "โมเดลเล็ก (laya-thai-distill-run3, 322M encoder, ~39 ms) ตอบก่อนทุกข้อ ข้อที่มันไม่มั่นใจ "
        f"(ความน่าจะเป็นสูงสุดต่ำกว่า {THRESHOLD:.2f}) ส่งต่อให้ OpenThai-SystemOne ที่ `{TEACHER_URL}` ตอบแทน "
        "request/response เหมือน `/v1/systemone` ของ teacher ทุกประการ เพิ่มเฉพาะ `usage.cascade` ที่บอกว่าข้อไหนถูกส่งต่อและเพราะอะไร "
        "วัดบน 2,455 decisions: แม่น 0.803 เทียบ teacher 0.816 โดยเรียก teacher 28% ของคำถาม"
    ),
    lifespan=lifespan,
)


@app.get("/healthz", tags=["ops"], summary="student โหลดแล้วและ teacher ตอบ /healthz หรือไม่")
def healthz():
    if not hasattr(app.state, "student"):
        raise HTTPException(status_code=503, detail="student not loaded")
    return {"ok": True, "student": STUDENT, "teacher": TEACHER_URL, "teacher_ok": teacher_healthy(),
            "threshold": THRESHOLD, "max_options": MAX_OPTIONS}


@app.get("/stats", tags=["ops"], summary="ตัวนับสะสม: request, คำถาม, สัดส่วนที่ไปถึง teacher, เวลาเฉลี่ย")
def stats():
    s = app.state.stats
    return {**s,
            "teacher_question_fraction": round(s["teacher_questions"] / s["questions"], 4) if s["questions"] else None,
            "student_ms_mean": round(s["student_ms_total"] / s["requests"], 1) if s["requests"] else None,
            "teacher_ms_mean": round(s["teacher_ms_total"] / s["teacher_calls"], 1) if s["teacher_calls"] else None,
            "config": {"threshold": THRESHOLD, "max_options": MAX_OPTIONS, "teacher": TEACHER_URL, "student": STUDENT}}


@app.post("/v1/systemone", tags=["decision"], summary="ตัดสินใจจาก state: student ก่อน, teacher เมื่อไม่มั่นใจ")
async def system_one(request: Request):
    body = await request.json()
    state = body.get("state")
    questions = body.get("questions")
    if state is None or not isinstance(questions, dict) or not questions:
        raise HTTPException(status_code=422, detail="need `state` and a non-empty `questions` object")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in QTYPES:
            raise HTTPException(status_code=422, detail="question %r: `type` must be one of %s" % (qid, list(QTYPES)))

    st = app.state.stats
    st["requests"] += 1
    st["questions"] += len(questions)

    t0 = time.perf_counter()
    student_error = None
    s_ans: Dict[str, Any] = {}
    input_tokens = 0
    try:
        s = await asyncio.to_thread(app.state.student.predict, state, questions)
        s_ans, input_tokens = s["answers"], int(s["usage"]["input_tokens"])
    except Exception as e:  # noqa: BLE001  (e.g. options exceed head_max_len -> everything to the teacher)
        student_error = str(e)
        st["student_errors"] += 1
    student_ms = (time.perf_counter() - t0) * 1000
    st["student_ms_total"] += student_ms

    answers: Dict[str, Any] = {}
    reasons: Dict[str, str] = {}
    for qid, q in questions.items():
        if qid in s_ans:
            why = route(q, s_ans[qid], student_confidence(q, s_ans[qid]))
            if why is None:
                answers[qid] = s_ans[qid]
                continue
            reasons[qid] = why
        else:
            reasons[qid] = "student_error"

    teacher_ms = None
    t_usage: Dict[str, Any] = {}
    if reasons:
        tbody = {k: body[k] for k in ("model", "order_invariant", "permutations") if k in body}
        tbody.update(state=state, questions={qid: questions[qid] for qid in reasons})
        t1 = time.perf_counter()
        st["teacher_calls"] += 1
        st["teacher_questions"] += len(reasons)
        try:
            t = await asyncio.to_thread(teacher_call, tbody)
            answers.update(t["answers"])
            t_usage = t.get("usage", {})
        except HTTPException:
            raise
        except Exception as e:  # noqa: BLE001
            st["teacher_failures"] += 1
            missing = [qid for qid in reasons if qid not in s_ans]
            if missing:
                raise HTTPException(status_code=503, detail="teacher unavailable and student could not answer %s: %s" % (missing, e))
            for qid in reasons:  # degrade to the student's low-confidence answer rather than fail
                answers[qid] = s_ans[qid]
                reasons[qid] += ";teacher_unavailable"
        teacher_ms = (time.perf_counter() - t1) * 1000
        st["teacher_ms_total"] += teacher_ms

    return {
        "model": body.get("model") or MODEL_NAME,
        "answers": {qid: answers[qid] for qid in questions},
        "usage": {
            "input_tokens": input_tokens + int(t_usage.get("input_tokens", 0)),
            "output_tokens": 0,
            "permutations": int(t_usage.get("permutations", 1)),
            "truncated": bool(t_usage.get("truncated", False)),
            "cascade": {
                "threshold": THRESHOLD, "max_options": MAX_OPTIONS,
                "teacher_questions": list(reasons), "reasons": reasons,
                "student_ms": round(student_ms, 1), "teacher_ms": round(teacher_ms, 1) if teacher_ms is not None else None,
                "student_error": student_error,
            },
        },
    }
