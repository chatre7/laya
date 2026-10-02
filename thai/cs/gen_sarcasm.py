"""Sarcasm the way Thai customers write it, built so the label is right by construction.

A first attempt asked the LLM for whole sarcastic messages: many read as sincere praise. So the LLM (Qwen3-8B on vLLM)
only writes EVENTS for each service topic, which it does well: things that went wrong, the same said with "แค่ ... เอง",
and things that went well. The praise is ours, the same phrases on both sides, in the patterns seen in real 1-star reviews
("แอพดีมากครับ เอาไป 1 ดาวพอ", "มาปิดปรับปรุงตอนจะใช้เงิน ดีจริงๆ", "เติมปุ๊บเงินหายปั๊บ เยี่ยมจริงๆ"):

  sarcastic      bad event + praise / praise + minimised bad event / "ขอบคุณที่ ..."   -> negative, frustration 1
  sincere        good event + the same praise                                          -> positive, frustration 0
  plain_negative bad event (+ a plain complaint)                                       -> negative
  indirect_praise written by the LLM ("ตอนแรกไม่คิดว่าจะดี แต่ ...")                      -> positive, frustration 0

The polarity is carried by the event, never by the praise word. Topics = the merged intent groups (intent_groups.py).

    python3 thai/cs/gen_sarcasm.py --out thai/data/cs/sarcasm_gen.jsonl
"""
import argparse
import json
import os
import random
import re
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intent_groups import GROUPS  # noqa: E402

BIZ_TH = {"telecom": "ค่ายมือถือ/อินเทอร์เน็ต", "banking": "ธนาคาร/แอปการเงิน", "insurance": "บริษัทประกัน"}
PRAISE = ["ดีจริง ๆ", "ดีจริงๆ ครับ", "เยี่ยมมากครับ", "เยี่ยมจริง ๆ ค่ะ", "บริการดีมากค่ะ", "บริการเยี่ยมมาก", "ประทับใจมากครับ", "ประทับใจสุด ๆ",
          "สุดยอดไปเลย", "สุดยอดมากค่ะ", "ดีมากเลยครับ", "ดีมากค่ะ", "เก่งมากค่ะ", "ยอดเยี่ยมจริง ๆ", "ดีงามมาก", "เริ่ดมากค่ะ", "ปังมาก",
          "ขอชื่นชมเลยครับ", "น่าชื่นชมจริง ๆ", "เลิศมากค่ะ", "ชอบมากเลย", "ถูกใจมากครับ", "แจ่มมาก", "ดีเกินคาดเลยค่ะ"]
PRAISE_OPEN = ["แอปดีมากครับ", "บริการดีมากค่ะ", "ระบบเสถียรมากค่ะ", "เร็วมากเลยครับ", "ประทับใจมากค่ะ", "สะดวกมากเลย", "ดูแลลูกค้าดีมากครับ",
               "ใส่ใจลูกค้ามากค่ะ", "ทำงานไวมากครับ", "คุณภาพดีมาก", "ใช้ง่ายมากค่ะ", "เยี่ยมไปเลยครับ"]
SARCASM_TAIL = ["เอาไป 1 ดาวพอ", "ยอมใจเลย 555", "สมกับเป็นบริษัทใหญ่จริง ๆ", "ปรบมือให้เลยค่ะ", "ขอบคุณมากนะคะ", "คนอื่นทำไม่ได้แบบนี้นะ", "ไม่ผิดหวังเลยจริง ๆ"]
COMPLAINT = ["แย่มาก", "ไม่ไหวเลย", "ช่วยแก้ไขด้วยครับ", "ผิดหวังมากค่ะ", "เซ็งมาก", "ไม่โอเคเลยค่ะ", "น่าเบื่อมาก", "ปรับปรุงด่วนเลยครับ", "", "", ""]
GOOD_TAIL = ["ขอบคุณมากค่ะ", "จะใช้ต่อไปแน่นอนครับ", "แนะนำเลยค่ะ", "ให้ 5 ดาวเลย", "", "", ""]

EVENT_PROMPT = """เขียนเหตุการณ์สั้น ๆ ที่ลูกค้าของ{biz}เจอ ในหัวข้อ: {topic}
ให้เขียนเป็นข้อเท็จจริงแบบที่ลูกค้าเล่า มีรายละเอียดเฉพาะ (ตัวเลข เวลา จำนวนครั้ง จำนวนเงิน) ยาว 5-18 คำ ไม่มีคำชม ไม่มีคำด่า ไม่มีคำแสดงอารมณ์ ไม่ต้องมีคำลงท้าย
- bad: 10 เหตุการณ์ที่แย่ เช่น "รอสายมา 40 นาทีแล้วยังไม่มีคนรับ", "โอนเงินไปสองวันแล้วยังไม่เข้า"
- bad_minimised: 10 เหตุการณ์ที่แย่ เขียนด้วยรูป "แค่ ... เอง" เช่น "รอสายแค่ 40 นาทีเอง", "เด้งออกแค่ 5 รอบเอง", "หักเงินซ้ำแค่ 2 ครั้งเอง"
- bad_action: 10 สิ่งแย่ที่บริษัททำ ขึ้นต้นด้วยคำกริยา ใช้ต่อท้ายคำว่า "ขอบคุณที่" ได้ เช่น "หักเงินซ้ำสองรอบ", "ปล่อยให้รอสายเกือบชั่วโมง"
- good: 10 เหตุการณ์ที่ดี เช่น "โทรไปไม่ถึงนาทีก็มีคนรับ", "เงินคืนเข้าบัญชีภายในวันเดียว"
- indirect_praise: 8 ข้อความชมแบบอ้อม 1-2 ประโยค ที่ลูกค้าพอใจจริง เช่น ตอนแรกไม่คาดหวังแต่ผลออกมาดี, ติเล็กน้อยแต่โดยรวมพอใจ, เทียบกับที่อื่นแล้วที่นี่ดีกว่า
ทุกข้อต้องต่างกัน และเกี่ยวกับหัวข้อนี้ ตอบเป็น JSON เท่านั้น"""
LISTS = {"bad": 10, "bad_minimised": 10, "bad_action": 10, "good": 10, "indirect_praise": 8}
SCHEMA = {"type": "object", "properties": {k: {"type": "array", "minItems": n, "maxItems": n, "items": {"type": "string"}} for k, n in LISTS.items()},
          "required": list(LISTS), "additionalProperties": False}


def ask(url, model, content, seed):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.9, "top_p": 0.95, "max_tokens": 1500, "seed": seed,
            "response_format": {"type": "json_schema", "json_schema": {"name": "events", "schema": SCHEMA, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(json.load(r)["choices"][0]["message"]["content"])
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)
    return {}


def clean(t):
    t = " ".join(str(t).split()).strip('"“”\'.,')
    thai = sum(1 for ch in t if "฀" <= ch <= "๿")
    alpha = sum(1 for ch in t if ch.isalpha())
    return t if 6 <= len(t) <= 160 and thai >= 0.6 * max(1, alpha) else ""


EMOTION = re.compile(r"ดีมาก|เยี่ยม|ประทับใจ|สุดยอด|แย่|ห่วย|กาก|เซ็ง|โกรธ|ผิดหวัง|พอใจ|ชอบ|ขอบคุณ")
# Read-through of the first output (show_sarcasm_gen.py): the LLM's "minimised" bad events are sometimes not bad ("ยอดใช้งานแค่
# 15GB เอง"), some "good" events are bad ("ไม่มีการแจ้งเตือนก่อนการหักเงิน"), and some indirect praise is a suggestion. Keep only
# the rows whose event is unmistakable; a wrong label here would teach the opposite of the point.
BAD = re.compile(r"ช้า|ผิด|ไม่|หาย|ค้าง|เด้ง|ล่ม|หัก|ซ้ำ|รอ|หลุด|ล้มเหลว|เกิน|ตัด|เสีย|ปฏิเสธ|ยกเลิก")
OK_NEG = re.compile(r"ไม่ต้อง|ไม่ถึง|ไม่มีปัญหา|ไม่ผิด|ไม่ถูกหัก|ไม่ล่ม|ไม่ค้าง|ไม่หลุด|ไม่เกิน|ไม่ติดขัด|ไม่สะดุด|ไม่เสีย|ไม่ช้า|ไม่ยุ่งยาก|ไม่ซับซ้อน")
NOT_GOOD = re.compile(r"ไม่|ช้า|ผิดพลาด|ล้มเหลว|ค้าง|หาย|ซ้ำ|เกิน")
POS = re.compile(r"ดี|พอใจ|ประทับใจ|ชอบ|เร็ว|สะดวก|ง่าย|สุภาพ|ชัดเจน|ทันที|คุ้ม|ไว|ราบรื่น")
NOT_PRAISE = re.compile(r"ต้องการให้|อยากให้|ควร|พอใช้|น่าจะ|ถ้า")


def keep(row):
    t = row["text"]
    if row["kind"] == "sarcastic":
        opener = next((p for p in PRAISE_OPEN if t.startswith(p)), None)
        return bool(BAD.search(t[len(opener):])) if opener else True  # praise + "แค่ ... เอง": the event must be clearly bad
    if row["kind"] == "sincere":
        return not NOT_GOOD.search(OK_NEG.sub("", t))
    if row["kind"] == "indirect_praise":
        return bool(POS.search(t)) and not NOT_PRAISE.search(t)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--model", default="qwen3-8b")
    ap.add_argument("--out", default="thai/data/cs/sarcasm_gen.jsonl")
    ap.add_argument("--calls", type=int, default=3, help="LLM calls per topic (10 events of each kind per call)")
    ap.add_argument("--per-topic", type=int, default=60, help="sarcastic and sincere messages composed per topic")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refilter", default="", help="apply keep() to an existing output file and write it to --out (no LLM)")
    args = ap.parse_args()
    if args.refilter:
        old = [json.loads(l) for l in open(args.refilter, encoding="utf-8")]
        new = [r for r in old if keep(r)]
        with open(args.out, "w", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(old)} {dict(Counter(r['kind'] for r in old))} -> {len(new)} {dict(Counter(r['kind'] for r in new))}")
        return
    rng = random.Random(args.seed)
    jobs = [(biz, g, EVENT_PROMPT.format(biz=BIZ_TH[biz], topic=desc), rng.randrange(10**9))
            for biz, gs in GROUPS.items() for g, (desc, _) in gs.items() for _ in range(args.calls)]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(lambda j: ask(args.url, args.model, j[2], j[3]), jobs))
    events = {}
    for (biz, g, _, _), o in zip(jobs, outs):
        e = events.setdefault((biz, g), {k: [] for k in LISTS})
        for k in LISTS:
            for t in o.get(k, []):
                t = clean(t)
                # events must be bare facts: an event that already carries praise or anger would blur the contrast
                if t and (k == "indirect_praise" or not EMOTION.search(t)) and t not in e[k]:
                    e[k].append(t)
    print(f"{len(jobs)} calls in {(time.time() - t0) / 60:.1f} min; events per topic: " +
          ", ".join(f"{k} {sum(len(e[k]) for e in events.values()) / len(events):.0f}" for k in LISTS), flush=True)

    rows, seen = [], set()

    def add(biz, g, kind, sentiment, frustration, text):
        text = " ".join(text.split())
        if text not in seen:
            seen.add(text)
            rows.append({"business": biz, "topic": g, "kind": kind, "sentiment": sentiment, "frustration": frustration, "text": text})

    for (biz, g), e in events.items():
        for _ in range(args.per_topic):
            form = rng.random()
            if e["bad"] and form < 0.45:
                tail = rng.choice(PRAISE) if rng.random() < 0.8 else rng.choice(SARCASM_TAIL)
                add(biz, g, "sarcastic", "negative", 1, f"{rng.choice(e['bad'])} {tail}")
            elif e["bad_minimised"] and form < 0.8:
                add(biz, g, "sarcastic", "negative", 1, f"{rng.choice(PRAISE_OPEN)} {rng.choice(e['bad_minimised'])}")
            elif e["bad_action"]:
                add(biz, g, "sarcastic", "negative", 1, f"ขอบคุณที่{rng.choice(e['bad_action'])}นะ{rng.choice(['คะ', 'ครับ'])} {rng.choice(PRAISE)}")
            if e["good"]:
                if rng.random() < 0.6:
                    add(biz, g, "sincere", "positive", 0, f"{rng.choice(e['good'])} {rng.choice(PRAISE)} {rng.choice(GOOD_TAIL)}")
                else:
                    add(biz, g, "sincere", "positive", 0, f"{rng.choice(PRAISE_OPEN)} {rng.choice(e['good'])}")
        for t in e["bad"]:
            add(biz, g, "plain_negative", "negative", None, f"{t} {rng.choice(COMPLAINT)}")
        for t in e["indirect_praise"]:
            add(biz, g, "indirect_praise", "positive", 0, t)
    rows = [r for r in rows if keep(r)]
    rng.shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} messages {dict(Counter(r['kind'] for r in rows))} -> {args.out}")


if __name__ == "__main__":
    main()
