"""Rewrite formal Thai customer-support utterances into the way real callers/chatters write, with an LLM behind an
OpenAI-compatible endpoint (vLLM). Keeps the intent label; produces N variants per input in different registers.

    python rewrite_colloquial.py --inp data/cc/porameht.jsonl --out data/cc/porameht_colloquial.jsonl --n 100 --variants 3
"""
import argparse
import json
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

STYLES = [
    "ลูกค้าพิมพ์แชทแบบรีบ ๆ สั้น ห้วน ไม่มีคำลงท้าย อาจพิมพ์ผิดหรือย่อคำ",
    "ลูกค้าโทรมาแล้วถูกถอดเสียงเป็นข้อความ พูดยาวหน่อย มีคำว่า ครับ/ค่ะ/นะ/อ่ะ/คือ แทรก มีการเล่าเหตุการณ์ก่อนบอกความต้องการ",
    "ลูกค้าหงุดหงิดหรือโกรธ บ่นก่อนแล้วค่อยบอกว่าต้องการอะไร",
    "ลูกค้าสุภาพมาก ถามอ้อม ๆ ไม่แน่ใจว่าต้องทำยังไง",
    "ลูกค้าวัยรุ่น ใช้ภาษาโซเชียล มีอีโมจิหรือคำแสลงบ้าง",
]
PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")
FILLERS = {
    "{{หมายเลขคำสั่งซื้อ}}": ["#48213", "ORD-20931", "เลขออเดอร์ 1187", "ออเดอร์ 7734"], "{{Order Number}}": ["#48213"], "{{เลขใบสั่งของ}}": ["#48213"],
    "{{ประเภทบัญชี}}": ["แบบพรีเมียม", "แบบฟรี", "แบบโปร"], "{{หมวดหมู่บัญชี}}": ["แบบพรีเมียม", "แบบฟรี"],
    "{{ชื่อบุคคล}}": ["คุณสมชาย", "พี่แนน", "คุณวิภา", "คุณต้น"], "{{จำนวนเงินคืน}}": ["1,290 บาท", "350 บาท", "2,000 บาท"], "{{จำนวนเงิน}}": ["1,290 บาท"],
    "{{สัญลักษณ์สกุลเงิน}}": ["", "฿"], "{{ประเทศที่จัดส่ง}}": ["ไทย", "ญี่ปุ่น", "สิงคโปร์"], "{{เมืองจัดส่ง}}": ["เชียงใหม่", "ขอนแก่น", "หาดใหญ่"],
    "{{Delivery City}}": ["เชียงใหม่"], "{{เมืองที่จัดส่ง}}": ["ภูเก็ต"], "{{หมายเลขใบแจ้งหนี้}}": ["INV-2291"], "{{ที่อยู่}}": ["ที่อยู่ใหม่"],
    "{{ชื่อผู้ใช้}}": ["ชื่อผู้ใช้"], "{{ที่อยู่อีเมล}}": ["อีเมล"], "{{อีเมล}}": ["อีเมล"], "{{หมายเลขโทรศัพท์}}": ["เบอร์"],
}
SPEAKERS = [("ผู้ชาย", "ผม", "ครับ"), ("ผู้หญิง", "ฉัน หรือ เรา", "ค่ะ/คะ"), ("ผู้หญิง", "หนู", "ค่ะ")]


def fill(text: str) -> str:
    def sub(m):
        return random.choice(FILLERS.get(m.group(0), ["อันนั้น"]))
    return PLACEHOLDER.sub(sub, text)


def prompt(text: str, intent: str, style: str, speaker) -> str:
    gender, pronoun, particle = speaker
    return (
        "เขียนข้อความของลูกค้าที่ติดต่อฝ่ายบริการลูกค้าใหม่ ให้เป็นภาษาไทยแบบที่คนไทยพิมพ์หรือพูดจริง\n"
        f"ความต้องการของลูกค้า (ห้ามเปลี่ยน): {intent}\n"
        f"ข้อความต้นฉบับ (ภาษาเขียน): {text}\n"
        f"ผู้พูดเป็น{gender} ใช้สรรพนาม \"{pronoun}\" และคำลงท้าย \"{particle}\" ให้สม่ำเสมอทั้งข้อความ (หรือไม่ใส่สรรพนาม/คำลงท้ายเลยก็ได้ แต่ห้ามปน)\n"
        f"ลักษณะข้อความ: {style}\n"
        "กติกา: ความหมายและความต้องการเหมือนเดิม, พูดในฐานะลูกค้าเท่านั้น, ห้ามใส่ {{...}}, ห้ามพิมพ์คำอธิบายลักษณะข้อความซ้ำ, "
        "ห้ามใช้ภาษาอื่นนอกจากไทย (ยกเว้นชื่อสินค้าหรือเลขออเดอร์), ยาว 1-3 ประโยค, ตอบเป็นข้อความลูกค้าอย่างเดียว ไม่ต้องอธิบาย"
    )


def chat(url, model, content, timeout=120):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.8, "top_p": 0.95, "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)["choices"][0]["message"]["content"].strip()
                out = re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip('"“”')
                return out
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1 + attempt)
    return "ERROR: %r" % (last,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100, help="how many source rows to sample (0 = all)")
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    for r in rows:
        r["intent"] = r["intent"].strip()
        r["category"] = r["category"].strip()
    if args.n:
        rows = random.sample(rows, args.n)
    jobs = []
    for i, r in enumerate(rows):
        for v in range(args.variants):
            style = STYLES[(i + v) % len(STYLES)]
            jobs.append((i, v, style, prompt(fill(r["text"]), r["intent"], style, random.choice(SPEAKERS))))
    t = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(lambda j: chat(args.url, args.model, j[3]), jobs))
    dt = time.perf_counter() - t
    def bad(o):
        return (o.startswith("ERROR") or re.search(r"(.)\1{6,}", o) is not None or len(o) > 400 or len(o) < 6
                or sum(1 for ch in o if "฀" <= ch <= "๿") < 0.5 * sum(1 for ch in o if ch.isalpha())
                or "ลักษณะข้อความ" in o or "ถอดเสียง" in o or "สรรพนาม" in o)
    outs = [("ERROR: filtered " + o[:40]) if (not o.startswith("ERROR") and bad(o)) else o for o in outs]
    n_err = sum(o.startswith("ERROR") for o in outs)
    with open(args.out, "w", encoding="utf-8") as f:
        for (i, v, style, _), o in zip(jobs, outs):
            r = rows[i]
            if o.startswith("ERROR"):
                continue
            f.write(json.dumps({"src": r["text"], "text": o, "category": r["category"], "intent": r["intent"], "style": style[:20], "variant": v},
                               ensure_ascii=False) + "\n")
    print(f"{len(jobs)} rewrites in {dt:.0f} s ({len(jobs)/dt:.1f}/s), {n_err} errors -> {args.out}")
    for (i, v, style, _), o in list(zip(jobs, outs))[: 5 * args.variants]:
        if v == 0:
            print(f"\n[{rows[i]['intent']}] {rows[i]['text']}")
        print(f"   ({style[:14]}...) {o}")


if __name__ == "__main__":
    main()
