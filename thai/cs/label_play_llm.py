"""Label Google Play reviews with an LLM (Qwen3-8B on vLLM) instead of the teacher: intent (that business's list or other),
department, urgency, as schema-constrained JSON. --eval scores the labeller on the 300 hand-labelled reviews first.

    python3 thai/cs/label_play_llm.py --eval                      # accuracy against data_domain/play_labels.json
    python3 thai/cs/label_play_llm.py --out thai/data/cs/play_llm.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs_questions import INTENTS, OTHER, SHARED  # noqa: E402

BIZ_TH = {"telecom": "ค่ายมือถือ/อินเทอร์เน็ต", "banking": "ธนาคาร/แอปการเงิน", "insurance": "บริษัทประกัน"}
DEPTS = SHARED["department"]["criteria"]
URG = SHARED["urgency"]["criteria"]
RULES = ("กติกา:\n"
         "- ถ้าข้อความเป็นแค่คำชม คำขอบคุณ คำบ่นลอย ๆ ที่ไม่บอกปัญหาหรือความต้องการ ข้อเสนอแนะฟีเจอร์ หรืออ่านไม่รู้เรื่อง ให้ intent = other, department = support, urgency = 0\n"
         "- เลือก intent ที่ตรงกับปัญหาหรือสิ่งที่ลูกค้าต้องการที่สุด ถ้าไม่มีข้อไหนตรงจริง ๆ ให้ตอบ other (อย่าเดา)\n"
         # added after the first pass on the 300 hand-labelled reviews (36 app problems were called other): the labelling convention
         "- ปัญหาการใช้แอป (เข้าไม่ได้ ค้าง ช้า เด้งออก ล่ม ปิดปรับปรุง อัปเดตแล้วใช้ไม่ได้ สมัครหรือลงทะเบียนในแอปไม่ได้ OTP ไม่มา) "
         "ถือเป็นปัญหาจริง ไม่ใช่คำบ่นลอย ๆ แม้จะเขียนสั้นหรือหยาบ: ธนาคาร/แอปการเงิน = app_or_login_problem, ค่ายมือถือ = report_problem, "
         "บริษัทประกัน (ไม่มี intent เรื่องแอป) = other แต่ department = technical และ urgency = 1\n"
         "- department: ปัญหาแอป/ระบบ/สัญญาณ/ใช้งานไม่ได้ = technical, เงินถูกหัก/ค่าธรรมเนียม/บิล = billing, บัญชี/บัตร/ข้อมูลส่วนตัว/สัญญา = account, "
         "อยากสมัคร/ซื้อ/ถามโปร = sales, เคลม/สินไหม = claims, ร้องเรียน/ติดต่อไม่ได้/ถามทั่วไป/คำชม = support\n"
         "- urgency: 0 = ไม่รีบ คำชม ถามทั่วไป ข้อเสนอแนะ, 1 = มีปัญหาที่ควรช่วยภายในวันนี้ (เช่น เข้าแอปไม่ได้ ใช้งานติดขัด), "
         "2 = ด่วน ลูกค้าเสียหายอยู่ (เงินหาย โอนแล้วเงินไม่เข้า เติมเงินแล้วไม่ได้ ใช้งานไม่ได้หลายวัน ขู่ฟ้อง)\n")


def prompt(biz, text):
    intents = "\n".join(f"- {k}: {v}" for k, v in INTENTS[biz].items()) + f"\n- other: {OTHER}"
    depts = "\n".join(f"- {k}: {v}" for k, v in DEPTS.items())
    return (f"คุณเป็นพนักงานคัดแยกเรื่องของ{BIZ_TH[biz]} อ่านข้อความของลูกค้าแล้วตอบเป็น JSON ที่มี intent, department, urgency\n\n"
            f"รายการ intent:\n{intents}\n\nรายการ department:\n{depts}\n\n{RULES}\nข้อความลูกค้า: {text}")


def schema(biz):
    return {"type": "object", "properties": {"intent": {"type": "string", "enum": list(INTENTS[biz]) + ["other"]},
                                             "department": {"type": "string", "enum": list(DEPTS)},
                                             "urgency": {"type": "integer", "enum": [0, 1, 2]}},
            "required": ["intent", "department", "urgency"], "additionalProperties": False}


def ask(url, model, biz, text):
    body = {"model": model, "messages": [{"role": "user", "content": prompt(biz, text[:600])}], "temperature": 0, "max_tokens": 60,
            "response_format": {"type": "json_schema", "json_schema": {"name": "label", "schema": schema(biz), "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                o = json.loads(json.load(r)["choices"][0]["message"]["content"])
            if o["intent"] in INTENTS[biz] or o["intent"] == "other":
                return [o["intent"], o["department"], int(o["urgency"])]
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--inp", default="thai/data_domain/play_reviews.jsonl")
    ap.add_argument("--out", default="thai/data/cs/play_llm.jsonl")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--sample", default="thai/data_domain/play_reviews_eval_sample.jsonl")
    ap.add_argument("--labels", default="thai/data_domain/play_labels.json")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.sample if args.eval else args.inp, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    t0 = time.time()
    done = [0]

    def work(r):
        o = ask(args.url, args.model, r["business"], r["text"])
        done[0] += 1
        if done[0] % 2000 == 0:
            print(f"  {done[0]}/{len(rows)}, {(time.time() - t0) / 60:.1f} min", flush=True)
        return o

    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(work, rows))
    print(f"{sum(o is not None for o in outs)}/{len(rows)} labelled in {(time.time() - t0) / 60:.1f} min", flush=True)
    if args.eval:
        gold = json.load(open(args.labels, encoding="utf-8"))["labels"]
        st, wrong = Counter(), Counter()
        for r, o in zip(rows, outs):
            g = gold[r["id"]]
            if o is None:
                o = ["other", "support", 0]
            for key in (r["business"], "all"):
                st[key, "n"] += 1
                st[key, "intent"] += o[0] == g[0]
                st[key, "department"] += o[1] == g[1]
                st[key, "urgency"] += o[2] == g[2]
                if g[0] != "other":
                    st[key, "n_in"] += 1
                    st[key, "false_other"] += o[0] == "other"
                else:
                    st[key, "n_other"] += 1
                    st[key, "other_caught"] += o[0] == "other"
            if o[0] != g[0]:
                wrong[f"{g[0]} -> {o[0]}"] += 1
        for key in ("telecom", "banking", "insurance", "all"):
            n = st[key, "n"]
            print(f"{key:10s} n={n:4d} intent {st[key, 'intent'] / n:.3f} dept {st[key, 'department'] / n:.3f} urg {st[key, 'urgency'] / n:.3f} "
                  f"false-other {st[key, 'false_other']}/{st[key, 'n_in']} other-caught {st[key, 'other_caught']}/{st[key, 'n_other']}")
        print("intent errors (gold -> llm):", wrong.most_common(15))
        return
    with open(args.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            if o is not None:
                f.write(json.dumps({**r, "labels": {"intent": o[0], "department": o[1], "urgency": o[2]}}, ensure_ascii=False) + "\n")
    print("intents", Counter(o[0] for o in outs if o).most_common(12), "departments", Counter(o[1] for o in outs if o), "urgency", Counter(o[2] for o in outs if o))


if __name__ == "__main__":
    main()
