"""Quality gate for LLM rewrites: ask the teacher (OpenThai-SystemOne) which of the 27 Bitext intents each rewritten
text expresses and whether it reads as a customer's message; report agreement with the source label per style.
Rows the teacher disagrees with are the ones to drop (or to fix the prompt for).

    python check_rewrites.py --inp data/cc/trial100.jsonl [--teacher http://172.18.72.145:8010] [--out data/cc/trial100_checked.jsonl]
"""
import argparse
import collections
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

INTENTS = {
    "cancel_order": "ขอยกเลิกคำสั่งซื้อ/ออเดอร์", "change_order": "ขอแก้ไข/เปลี่ยนแปลงรายการในคำสั่งซื้อ",
    "change_shipping_address": "ขอเปลี่ยนที่อยู่จัดส่งของออเดอร์ที่มีอยู่", "check_cancellation_fee": "ถามค่าธรรมเนียมการยกเลิก",
    "check_invoice": "ถามหรือขอตรวจสอบรายละเอียดใบแจ้งหนี้", "check_payment_methods": "ถามว่ามีวิธีชำระเงินแบบไหนบ้าง",
    "check_refund_policy": "ถามนโยบาย/เงื่อนไขการคืนเงิน", "complaint": "ร้องเรียน ไม่พอใจบริการหรือสินค้า ต้องการแจ้งปัญหา",
    "contact_customer_service": "ถามช่องทางติดต่อฝ่ายบริการลูกค้า (อีเมล เบอร์ เวลาทำการ)", "contact_human_agent": "ขอคุยกับพนักงานที่เป็นคนจริง",
    "create_account": "ขอสมัคร/เปิดบัญชีใหม่", "delete_account": "ขอลบ/ปิดบัญชี", "delivery_options": "ถามว่ามีตัวเลือกการจัดส่งแบบไหนบ้าง",
    "delivery_period": "ถามว่าจะได้รับสินค้าเมื่อไหร่ ใช้เวลาส่งกี่วัน", "edit_account": "ขอแก้ไขข้อมูลในบัญชี",
    "get_invoice": "ขอรับ/ดาวน์โหลดใบแจ้งหนี้", "get_refund": "ขอเงินคืน", "newsletter_subscription": "สมัคร/ยกเลิกรับจดหมายข่าว",
    "payment_issue": "ชำระเงินไม่ผ่าน มีปัญหาการจ่ายเงิน", "place_order": "ต้องการสั่งซื้อสินค้า", "recover_password": "ลืมรหัสผ่าน ขอกู้คืน",
    "registration_problems": "สมัครสมาชิกไม่สำเร็จ มีปัญหาตอนลงทะเบียน", "review": "ต้องการเขียนรีวิว/ให้ความเห็นเกี่ยวกับสินค้าหรือบริการ",
    "set_up_shipping_address": "ขอเพิ่ม/ตั้งค่าที่อยู่จัดส่งใหม่", "switch_account": "ขอเปลี่ยนประเภทบัญชี (เช่น ฟรี <-> พรีเมียม)",
    "track_order": "ขอติดตามสถานะออเดอร์ ของถึงไหนแล้ว", "track_refund": "ขอติดตามสถานะการคืนเงิน เงินคืนถึงไหนแล้ว",
}
QUESTIONS = {
    "intent": {"type": "choice", "instructions": "ลูกค้าต้องการอะไร", "criteria": INTENTS},
    "is_customer": {"type": "noul", "instructions": "ผู้เขียนข้อความนี้คือลูกค้าที่กำลังขอความช่วยเหลือใช่หรือไม่", "criteria": {"true": "ลูกค้าเขียนมาถาม/ขอ/บ่น", "false": "พนักงานหรือบริษัทเขียนตอบลูกค้า"}},
    "natural": {"type": "score", "instructions": "ข้อความนี้อ่านเป็นภาษาไทยที่คนจริงพิมพ์หรือพูดแค่ไหน",
                "criteria": ["แข็งเหมือนแปลจากภาษาอื่น", "พอใช้", "เป็นธรรมชาติเหมือนคนไทยพิมพ์เอง"]},
}


def ask(url, text, order_invariant=False):
    body = json.dumps({"state": text, "questions": QUESTIONS, "order_invariant": order_invariant}, ensure_ascii=False).encode()
    req = urllib.request.Request(url + "/v1/systemone", body, {"content-type": "application/json"})
    for _ in range(3):
        try:
            return json.load(urllib.request.urlopen(req, timeout=120))["answers"]
        except Exception:  # noqa: BLE001
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True)
    ap.add_argument("--out")
    ap.add_argument("--teacher", default="http://172.18.72.145:8010")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--order-invariant", action="store_true", help="average over option orders (slower, ~2x)")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    with ThreadPoolExecutor(args.workers) as ex:
        answers = list(ex.map(lambda r: ask(args.teacher, r["text"], args.order_invariant), rows))
    by_style = collections.defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0])
    bad = []
    for r, a in zip(rows, answers):
        if a is None:
            continue
        ok = a["intent"]["choice"] == r["intent"]
        r["teacher_intent"], r["teacher_p"], r["is_customer"], r["natural"] = a["intent"]["choice"], a["intent"]["probabilities"][r["intent"]], a["is_customer"]["noul"], a["natural"]["score"]
        r["abstain"] = a["intent"].get("abstain")
        s = by_style[r.get("style", "?")]
        s[0] += 1; s[1] += ok; s[2] += r["teacher_p"]; s[3] += r["is_customer"]; s[4] += r["natural"]
        if not ok or r["is_customer"] < 0.5:
            bad.append(r)
    n = sum(s[0] for s in by_style.values())
    print(f"{n} rows checked; intent agreement {sum(s[1] for s in by_style.values())/n:.3f}")
    print(f"{'style':22s} {'n':>4s} {'agree':>6s} {'p(label)':>9s} {'customer':>9s} {'natural':>8s}")
    for st, s in sorted(by_style.items()):
        print(f"{st:22s} {s[0]:4d} {s[1]/s[0]:6.3f} {s[2]/s[0]:9.3f} {s[3]/s[0]:9.3f} {s[4]/s[0]:8.2f}")
    print(f"\n{len(bad)} rows the teacher would drop (intent mismatch or not a customer message); first 12:")
    for r in bad[:12]:
        print(f"  [{r['intent']} -> {r['teacher_intent']} p={r['teacher_p']:.2f} cust={r['is_customer']:.2f}] {r['text'][:110]}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
