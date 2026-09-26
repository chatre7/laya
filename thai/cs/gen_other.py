"""Out-of-scope (`other`) examples in the same register as the in-domain training text: messages a person might type into a
customer-service chat that are NOT a service request, written by the same LLM with the same styles and speakers as
translate_colloquial.py, so `other` cannot be learned from register alone (the run 6 failure mode).

    python gen_other.py --out data/cs/other_indomain.jsonl --n 6000
"""
import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from translate_colloquial import SPEAKERS, STYLES, bad, chat  # noqa: E402

KINDS = {
    "greeting_only": "ทักทายหรือพิมพ์มาเปิดแชทเฉย ๆ ยังไม่บอกว่าต้องการอะไร (เช่น สวัสดี, มีใครอยู่ไหม, ทดสอบ)",
    "small_talk": "คุยเล่นเรื่องทั่วไป อากาศ อาหาร ชีวิตประจำวัน วันหยุด ไม่เกี่ยวกับบริการ",
    "general_question": "ถามคำถามความรู้ทั่วไป ข่าว กีฬา บันเทิง การเมือง สุขภาพ ที่ไม่เกี่ยวกับบริษัท",
    "absurd_request": "ขอสิ่งที่ไม่มีเหตุผลหรือล้อเล่น เช่น ขอส่วนลดเพราะดวง ขอให้พนักงานแต่งคอสเพลย์ ถามว่ามีพลังวิเศษไหม",
    "wrong_window": "พิมพ์ผิดหน้าต่าง เหมือนคุยกับเพื่อนหรือครอบครัว เช่น เดี๋ยวถึงบ้าน ซื้อข้าวให้ด้วย",
    "spam_or_ad": "สแปม โฆษณา ชวนลงทุน ชวนเล่นพนัน ส่งลิงก์",
    "meaningless": "ข้อความสั้นมากที่ไม่มีความหมาย เช่น อืม, ค่ะ, 555, ???, ok, สติ๊กเกอร์",
    "thanks_only": "ขอบคุณหรือลาอย่างเดียว หลังจากคุยเสร็จ",
    "feedback_general": "ชมหรือบ่นบริษัทลอย ๆ โดยไม่มีคำขออะไร เช่น บริษัทนี้ดีจัง / ห่วยมาก",
}
BUSINESS_TH = ["ค่ายมือถือ", "ธนาคาร", "บริษัทประกัน", "ร้านค้าออนไลน์"]


def prompt(kind, style, speaker, business):
    gender, pronoun, particle = speaker
    return (
        f"เขียนข้อความ 1 ข้อความที่คนคนหนึ่งพิมพ์เข้ามาในแชทฝ่ายบริการลูกค้าของ{business} แต่ข้อความนั้น **ไม่ใช่เรื่องขอความช่วยเหลือหรือคำถามเกี่ยวกับบริการ** "
        f"ลักษณะ: {KINDS[kind]}\n"
        f"ผู้พิมพ์เป็น{gender} ใช้สรรพนาม \"{pronoun}\" และคำลงท้าย \"{particle}\" ให้สม่ำเสมอ (หรือไม่ใส่เลยก็ได้ แต่ห้ามปน)\n"
        f"ลักษณะการพิมพ์: {style}\n"
        "กติกา: ห้ามพูดถึงแพ็กเกจ บิล บัตร บัญชี เคลม กรมธรรม์ ออเดอร์ สินค้า หรือปัญหาการใช้บริการใด ๆ, ภาษาไทยเท่านั้น, ยาว 1-2 ประโยค (หรือสั้นกว่านั้นถ้าลักษณะกำหนด), "
        "ตอบเป็นข้อความนั้นอย่างเดียว ไม่ต้องอธิบาย"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    random.seed(args.seed)
    kinds = list(KINDS)
    jobs = []
    for i in range(args.n):
        kind, style = kinds[i % len(kinds)], STYLES[(i // len(kinds)) % len(STYLES)]
        jobs.append((kind, style, prompt(kind, style, random.choice(SPEAKERS), random.choice(BUSINESS_TH))))
    t = time.perf_counter()
    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(lambda j: chat(args.url, args.model, j[2]), jobs))
    n_bad, seen = 0, set()
    with open(args.out, "w", encoding="utf-8") as f:
        for (kind, style, _), o in zip(jobs, outs):
            short_ok = kind == "meaningless" and 1 <= len(o) < 6 and not o.startswith("ERROR")  # allow very short ones for that kind
            if (bad(o) and not short_ok) or o in seen:
                n_bad += 1
                continue
            seen.add(o)
            f.write(json.dumps({"text": o, "kind": kind, "style": style[:20]}, ensure_ascii=False) + "\n")
    print(f"{len(jobs)} generations in {(time.perf_counter() - t) / 60:.1f} min, {n_bad} filtered/duplicate -> {args.out}", flush=True)
    for (kind, style, _), o in list(zip(jobs, outs))[:18]:
        print(f"   [{kind:16s}] {o[:110]}")


if __name__ == "__main__":
    main()
