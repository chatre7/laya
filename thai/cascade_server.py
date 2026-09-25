"""Student -> teacher cascade server: laya-thai-callcenter (run 5) answers first, OpenThai-SystemOne answers what it is unsure about.

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
import concurrent.futures
import json
import os
import queue
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
MODEL_NAME = os.environ.get("CASCADE_MODEL_NAME", "laya-thai-callcenter+openthai-systemone")
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
    """laya agent with dynamic batching: one thread owns the GPU. Requests are encoded on the caller's thread, queued,
    and the GPU thread collates everything that arrived within STUDENT_MAX_WAIT_MS (bounded by STUDENT_MAX_BATCH
    requests and STUDENT_MAX_BATCH_TOKENS padded tokens) into one forward, then decodes each request with laya's own
    formula (temperature buckets, confidence, act head), so a batched answer equals a single one up to bf16 noise.
    STUDENT_COMPILE=1 wraps the model in torch.compile (dynamic shapes)."""

    def __init__(self, path: str):
        import laya  # imported here so the module loads without torch (tests, docs)
        import torch
        from laya.common import QTYPES as qtypes, build_sequence, collate_items, render_options

        self.torch, self.build_sequence, self.collate_items, self.render_options, self.qtypes = torch, build_sequence, collate_items, render_options, qtypes
        self.agent = laya.Agent(path, device="cuda")
        self.cfg = dict(self.agent.cfg)
        self.max_batch = int(os.environ.get("STUDENT_MAX_BATCH", "32"))
        self.max_batch_tokens = int(os.environ.get("STUDENT_MAX_BATCH_TOKENS", "24576"))
        self.max_wait = float(os.environ.get("STUDENT_MAX_WAIT_MS", "6")) / 1000
        self.compiled = os.environ.get("STUDENT_COMPILE", "0") == "1"
        if self.compiled:
            self.agent.model = torch.compile(self.agent.model, dynamic=True)
        self.q: "queue.Queue" = queue.Queue()
        self.stats = {"student_forwards": 0, "student_max_batch": 0, "student_max_rows": 0, "student_batch_failures": 0}
        self.thread = threading.Thread(target=self._loop, name="student-gpu", daemon=True)
        self.thread.start()

    # -- encode on the caller's thread (CPU), same steps as Agent.system_one up to the forward
    def encode(self, state: Any, questions: Dict[str, Dict[str, Any]]):
        items, qs = [], []
        max_len, head_max_len = self.cfg.get("max_len", 512), self.cfg.get("head_max_len", 192)
        for qid, qd in questions.items():
            q = self.agent._to_internal(qd)
            seq, markers = self.build_sequence(self.agent.tok, state, q, max_len, head_max_len)
            if len(markers) != len(self.render_options(q)):
                raise ValueError("question %r options exceed head_max_len=%d" % (qid, head_max_len))
            items.append({"ids": seq, "markers": markers, "qtype": self.qtypes[q["t"]]})
            qs.append((qid, q))
        return items, qs

    def predict(self, state: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        items, qs = self.encode(state, questions)
        fut: "concurrent.futures.Future" = concurrent.futures.Future()
        self.q.put((items, qs, fut))
        return fut.result()

    # -- GPU thread: gather a batch, forward once, decode per request
    def _loop(self):
        while True:
            jobs = [self.q.get()]
            deadline = time.perf_counter() + self.max_wait
            rows = len(jobs[0][0])
            longest = max(len(it["ids"]) for it in jobs[0][0])
            while len(jobs) < self.max_batch:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    j = self.q.get(timeout=remaining)
                except queue.Empty:
                    break
                r, l = rows + len(j[0]), max(longest, max(len(it["ids"]) for it in j[0]))
                if r * l > self.max_batch_tokens:
                    self.q.put(j)  # over the padded-token budget: leave it for the next batch
                    break
                jobs.append(j)
                rows, longest = r, l
            try:
                outs = self._forward([j[0] for j in jobs])
                for (items, qs, fut), (logits, act, n_tok) in zip(jobs, outs):
                    fut.set_result(self._decode(items, qs, logits, act, n_tok))
            except Exception:  # noqa: BLE001  (e.g. OOM on a large batch): retry one by one
                self.stats["student_batch_failures"] += 1
                for items, qs, fut in jobs:
                    try:
                        logits, act, n_tok = self._forward([items])[0]
                        fut.set_result(self._decode(items, qs, logits, act, n_tok))
                    except Exception as e:  # noqa: BLE001
                        fut.set_exception(e)

    def _forward(self, groups):
        torch = self.torch
        b = self.collate_items(groups, self.agent.tok.pad_token_id)
        dev = self.agent.device
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=self.agent.dtype, enabled=dev.type == "cuda"):
            logits, act = self.agent.model(b["input_ids"].to(dev), b["attention_mask"].to(dev), b["marker_pos"].to(dev),
                                           b["marker_mask"].to(dev), b["qtype"].to(dev))
        logits = logits.float().cpu().numpy()
        act = torch.softmax(act.float(), -1).cpu().numpy()
        self.stats["student_forwards"] += 1
        self.stats["student_max_batch"] = max(self.stats["student_max_batch"], len(groups))
        self.stats["student_max_rows"] = max(self.stats["student_max_rows"], int(b["input_ids"].shape[0]))
        tokens = b["attention_mask"].sum(1).tolist()
        outs, r0 = [], 0
        for g in groups:
            r1 = r0 + len(g)
            outs.append((logits[r0:r1], act[r0:r1], int(sum(tokens[r0:r1]))))
            r0 = r1
        return outs

    def _decode(self, items, qs, logits, act, n_tokens):
        import numpy as np
        from laya.common import confidence_from_probs, temp_bucket

        ag = self.agent
        answers = {}
        for r, (qid, q) in enumerate(qs):
            k = len(items[r]["markers"])
            qt = self.qtypes[q["t"]]
            t_scale = ag.temperature_by_options.get(temp_bucket(qt, k), ag.temperature[qt])
            z = logits[r, :k] / t_scale
            p = np.exp(z - z.max())
            p = p / p.sum()
            conf = round(confidence_from_probs(p, k), 4)
            ext = {"act_probability": round(float(act[r, 0]), 4)}
            if q["t"] == "choice":
                keys = list(q["crit"].keys())
                answers[qid] = {"type": "choice", "choice": keys[int(p.argmax())],
                                "probabilities": {kk: round(float(v), 4) for kk, v in zip(keys, p)}, "confidence": conf, "action": ext}
            elif q["t"] == "score":
                answers[qid] = {"type": "score", "score": round(float((np.arange(k) * p).sum()), 4),
                                "legend": {str(i): c for i, c in enumerate(q["crit"])},
                                "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)}, "confidence": conf, "action": ext}
            else:
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4),
                                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4), "action": ext}
        return {"model": "laya-rl-agent", "answers": answers, "usage": {"input_tokens": n_tokens, "output_tokens": 0}}


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
    # asyncio.to_thread uses the loop's default executor; widen it so 64 requests can wait on the student/teacher at once
    asyncio.get_running_loop().set_default_executor(concurrent.futures.ThreadPoolExecutor(max_workers=64, thread_name_prefix="cascade"))
    app.state.student = Student(STUDENT)
    # warm-up over representative shapes (question counts, option counts, state lengths, batch sizes) so that torch.compile,
    # when enabled, does its compiling here rather than on the first live requests of each shape (several seconds each)
    warm = []
    for n_opt in (2, 4, 12, 30, 60):
        crit = {"opt%d" % i: "ตัวเลือกที่ %d สำหรับทดสอบ" % i for i in range(n_opt)}
        warm.append({"c": {"type": "choice", "instructions": "ข้อความนี้เข้าข่ายข้อใด", "criteria": crit}})
    for n_q in (1, 3, 5, 9):
        qs = {}
        for i in range(n_q):
            qs["q%d" % i] = [{"type": "noul", "instructions": "คำถามทดสอบที่ %d ใช่หรือไม่" % i},
                             {"type": "score", "instructions": "ระดับทดสอบ", "criteria": ["ต่ำ", "กลาง", "สูง"]},
                             {"type": "choice", "instructions": "หมวดทดสอบ", "criteria": {"a": "ก", "b": "ข", "c": "ค", "d": "ง", "e": "จ"}}][i % 3]
        warm.append(qs)
    states = ["ทดสอบระบบ", "โดนหักเงินซ้ำสองครั้งเมื่อวานนี้ ขอเงินคืนด่วนนะครับ ติดต่อไปแล้วยังไม่มีใครตอบเลย " * 3,
              "ลูกค้าโทรมาแจ้งว่าอินเทอร์เน็ตบ้านหลุดบ่อยมากตั้งแต่เมื่อวาน รีสตาร์ทเราเตอร์แล้วก็ยังเป็นอยู่ อยากให้ช่างเข้ามาดูภายในวันนี้ " * 12]
    t_w = time.perf_counter()
    for st_text in states:
        for qs in warm:
            app.state.student.predict(st_text, qs)
    with concurrent.futures.ThreadPoolExecutor(16) as ex:  # a few batched shapes too
        list(ex.map(lambda i: app.state.student.predict(states[i % 3], warm[i % len(warm)]), range(48)))
    print("[cascade] warm-up: %d shapes in %.0f s" % (len(states) * len(warm) + 48, time.perf_counter() - t_w), flush=True)
    app.state.stats = {"requests": 0, "questions": 0, "teacher_calls": 0, "teacher_questions": 0,
                       "student_errors": 0, "teacher_failures": 0, "student_ms_total": 0.0, "teacher_ms_total": 0.0}
    st = app.state.student
    print("[cascade] student %s loaded in %.1f s, head_max_len=%s, batching max %d req / %d tokens / %.0f ms, compile=%s, teacher %s, threshold %.2f, max_options %d"
          % (STUDENT, time.perf_counter() - t, st.cfg.get("head_max_len"), st.max_batch, st.max_batch_tokens, st.max_wait * 1000, st.compiled,
             TEACHER_URL, THRESHOLD, MAX_OPTIONS), flush=True)
    yield


app = FastAPI(
    title="laya -> OpenThai-SystemOne cascade",
    version="0.1.0",
    description=(
        "โมเดลเล็ก (laya-thai-callcenter run 5, 322M encoder, ~39 ms) ตอบก่อนทุกข้อ ข้อที่มันไม่มั่นใจ "
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
    if not app.state.student.thread.is_alive():
        raise HTTPException(status_code=503, detail="student gpu thread is dead")
    return {"ok": True, "student": STUDENT, "teacher": TEACHER_URL, "teacher_ok": teacher_healthy(),
            "threshold": THRESHOLD, "max_options": MAX_OPTIONS}


@app.get("/stats", tags=["ops"], summary="ตัวนับสะสม: request, คำถาม, สัดส่วนที่ไปถึง teacher, เวลาเฉลี่ย")
def stats():
    s = {**app.state.stats, **app.state.student.stats}
    return {**s,
            "teacher_question_fraction": round(s["teacher_questions"] / s["questions"], 4) if s["questions"] else None,
            "student_ms_mean": round(s["student_ms_total"] / s["requests"], 1) if s["requests"] else None,
            "teacher_ms_mean": round(s["teacher_ms_total"] / s["teacher_calls"], 1) if s["teacher_calls"] else None,
            "config": {"threshold": THRESHOLD, "max_options": MAX_OPTIONS, "teacher": TEACHER_URL, "student": STUDENT,
                       "student_max_batch": app.state.student.max_batch, "student_max_batch_tokens": app.state.student.max_batch_tokens,
                       "student_max_wait_ms": app.state.student.max_wait * 1000, "student_compile": app.state.student.compiled}}


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
