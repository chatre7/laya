"""English Bitext utterance (telco / banking / insurance) -> Thai customer message the way real callers/chatters write it,
via an OpenAI-compatible LLM endpoint (vLLM). Keeps business/category/intent; N variants in different registers with a
fixed speaker; placeholders filled with plausible Thai values; degenerate outputs filtered.

    python translate_colloquial.py --inp data/cs/bitext_cs_en.jsonl --out data/cs/cs_colloquial.jsonl --variants 2
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

STYLES = [
    "ลูกค้าพิมพ์แชทแบบรีบ ๆ สั้น ห้วน ไม่มีคำลงท้าย อาจพิมพ์ผิดหรือย่อคำ",
    "ลูกค้าโทรมาแล้วถูกบันทึกเป็นข้อความ พูดยาวหน่อย มีคำว่า ครับ/ค่ะ/นะ/อ่ะ/คือ แทรก เล่าเหตุการณ์ก่อนบอกความต้องการ",
    "ลูกค้าหงุดหงิดหรือโกรธ บ่นก่อนแล้วค่อยบอกว่าต้องการอะไร",
    "ลูกค้าสุภาพมาก ถามอ้อม ๆ ไม่แน่ใจว่าต้องทำยังไง",
    "ลูกค้าวัยรุ่น ใช้ภาษาโซเชียล มีอีโมจิหรือคำแสลงบ้าง",
]
SPEAKERS = [("ผู้ชาย", "ผม", "ครับ"), ("ผู้หญิง", "ฉัน หรือ เรา", "ค่ะ/คะ"), ("ผู้หญิง", "หนู", "ค่ะ")]
BUSINESS_TH = {"telecom": "ค่ายมือถือ/อินเทอร์เน็ต (เช่น แพ็กเกจ ซิม สัญญาณ ค่าบริการ โรมมิ่ง)",
               "banking": "ธนาคาร (เช่น บัตร บัญชี โอนเงิน สินเชื่อ ตู้ ATM รหัสผ่าน)",
               "insurance": "บริษัทประกัน (เช่น กรมธรรม์ เคลม ความคุ้มครอง เบี้ย ต่ออายุ)"}
PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")
FILLERS = {"Order Number": ["#48213", "ORD-20931"], "Invoice Number": ["INV-2291", "เลขที่ 004512"], "Account Number": ["xxx-x-x1234-x"],
           "Card Number": ["ลงท้าย 4482"], "Phone Number": ["08x-xxx-1234"], "Policy Number": ["กรมธรรม์ P-30291"], "Claim Number": ["เคลม C-1187"],
           "Amount": ["1,290 บาท", "350 บาท"], "Currency Symbol": [""], "Date": ["วันที่ 15", "สิ้นเดือน"], "Person Name": ["คุณสมชาย", "คุณวิภา"],
           "Customer Support Phone Number": ["1234"], "Company": ["บริษัท"], "Website URL": ["เว็บไซต์"], "Plan": ["แพ็กเกจ 599"],
           "Delivery City": ["เชียงใหม่"], "Delivery Country": ["ไทย"], "Account Type": ["แบบพรีเมียม"], "Account Category": ["แบบพรีเมียม"],
           "Salutation": [""], "Client First Name": ["สมชาย"], "Client Last Name": [""], "Refund Amount": ["1,290 บาท"], "Money Amount": ["2,000 บาท"]}


def fill(text):
    return PLACEHOLDER.sub(lambda m: random.choice(FILLERS.get(m.group(0)[2:-2].strip(), ["อันนั้น"])), text)


def prompt(text, business, category, intent, style, speaker):
    gender, pronoun, particle = speaker
    return (
        "แปลและเขียนใหม่: ข้อความลูกค้าภาษาอังกฤษด้านล่างที่ติดต่อ" + BUSINESS_TH[business] + " ให้เป็นข้อความภาษาไทยแบบที่คนไทยพิมพ์หรือพูดจริงกับฝ่ายบริการลูกค้า\n"
        f"ความต้องการของลูกค้า (ห้ามเปลี่ยน): {intent} (หมวด {category})\n"
        f"ข้อความต้นฉบับ: {text}\n"
        f"ผู้พูดเป็น{gender} ใช้สรรพนาม \"{pronoun}\" และคำลงท้าย \"{particle}\" ให้สม่ำเสมอ (หรือไม่ใส่เลยก็ได้ แต่ห้ามปน)\n"
        f"ลักษณะข้อความ: {style}\n"
        "กติกา: ความหมายและความต้องการเหมือนเดิม, ใช้คำที่คนไทยใช้จริงกับบริการนี้ (เช่น แพ็กเกจ ซิม บัตร บัญชี เคลม กรมธรรม์), พูดในฐานะลูกค้าเท่านั้น, "
        "ห้ามใส่ {{...}}, ห้ามพิมพ์คำอธิบายลักษณะข้อความซ้ำ, ห้ามใช้ภาษาอื่นนอกจากไทย (ยกเว้นชื่อบริการหรือเลขอ้างอิง), ยาว 1-3 ประโยค, "
        "ตอบเป็นข้อความลูกค้าอย่างเดียว ไม่ต้องอธิบาย"
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
                return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip('"“”')
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1 + attempt)
    return "ERROR: %r" % (last,)


def bad(o):
    thai = sum(1 for ch in o if "฀" <= ch <= "๿")
    alpha = sum(1 for ch in o if ch.isalpha())
    return (o.startswith("ERROR") or re.search(r"(.)\1{6,}", o) is not None or len(o) > 400 or len(o) < 6
            or thai < 0.6 * max(1, alpha) or "ลักษณะข้อความ" in o or "ต้นฉบับ" in o or "สรรพนาม" in o or "{{" in o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=0, help="sample this many source rows (0 = all)")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    if args.n:
        rows = random.sample(rows, args.n)
    jobs = []
    for i, r in enumerate(rows):
        for v in range(args.variants):
            style = STYLES[(i + v) % len(STYLES)]
            jobs.append((i, v, style, prompt(fill(r["text_en"]), r["business"], r["category"], r["intent"], style, random.choice(SPEAKERS))))
    t = time.perf_counter()
    done = [0]

    def work(j):
        o = chat(args.url, args.model, j[3])
        done[0] += 1
        if done[0] % 2000 == 0:
            el = time.perf_counter() - t
            print(f"  {done[0]}/{len(jobs)} in {el / 60:.1f} min ({done[0] / el:.1f}/s)", flush=True)
        return o

    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(work, jobs))
    n_bad = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for (i, v, style, _), o in zip(jobs, outs):
            if bad(o):
                n_bad += 1
                continue
            r = rows[i]
            f.write(json.dumps({"text": o, "text_en": r["text_en"], "business": r["business"], "category": r["category"], "intent": r["intent"],
                                "style": style[:20], "variant": v}, ensure_ascii=False) + "\n")
    print(f"{len(jobs)} generations in {(time.perf_counter() - t) / 60:.1f} min, {n_bad} filtered -> {args.out}", flush=True)
    for (i, v, style, _), o in list(zip(jobs, outs))[: 4 * args.variants]:
        if v == 0:
            print(f"\n[{rows[i]['business']}/{rows[i]['intent']}] {rows[i]['text_en'][:80]}")
        print(f"   ({style[:12]}...) {o[:140]}")


if __name__ == "__main__":
    sys.exit(main())
