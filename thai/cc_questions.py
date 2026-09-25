"""The call-center question set (runs 4-5): the contract the student is trained for. No heavy imports, so the review and
scoring tools can use it on any machine. `label_cc.py` imports it too."""
from check_rewrites import INTENTS

CATEGORIES = {
    "ORDER": "สั่งซื้อ/แก้ไขคำสั่งซื้อ", "SHIPPING": "ที่อยู่จัดส่ง", "CANCEL": "ยกเลิก ค่าธรรมเนียมยกเลิก", "INVOICE": "ใบแจ้งหนี้",
    "PAYMENT": "การชำระเงิน", "REFUND": "การคืนเงิน", "FEEDBACK": "ร้องเรียน รีวิว", "CONTACT": "ช่องทางติดต่อ ขอคุยกับคน",
    "ACCOUNT": "บัญชีผู้ใช้ รหัสผ่าน สมัครสมาชิก", "DELIVERY": "ตัวเลือกและระยะเวลาการจัดส่ง", "SUBSCRIPTION": "จดหมายข่าว การสมัครรับข้อมูล",
}
OTHER_INTENT = "ไม่เข้าข่ายข้อใดข้างต้น หรือไม่ใช่เรื่องติดต่อฝ่ายบริการลูกค้า"
QUESTIONS = {
    "intent": {"type": "choice", "instructions": "ลูกค้าต้องการอะไร", "criteria": dict(INTENTS)},
    "category": {"type": "choice", "instructions": "เรื่องที่ลูกค้าติดต่ออยู่ในหมวดใด", "criteria": CATEGORIES},
    "department": {"type": "choice", "instructions": "ควรส่งเรื่องนี้ให้ทีมใดรับผิดชอบ",
                   "criteria": {"billing": "ค่าบริการ ใบแจ้งหนี้ การชำระเงิน การคืนเงิน", "shipping": "การจัดส่ง ที่อยู่ ติดตามพัสดุ",
                                "account": "บัญชีผู้ใช้ รหัสผ่าน สมัคร/ลบบัญชี", "sales": "สั่งซื้อ เปลี่ยนแพ็กเกจ สมัครบริการ",
                                "support": "ปัญหาการใช้งาน ร้องเรียน ขอคุยกับพนักงาน เรื่องทั่วไป"}},
    "urgency": {"type": "score", "instructions": "เรื่องนี้เร่งด่วนแค่ไหน",
                "criteria": ["ไม่รีบ ถามข้อมูลทั่วไป", "ควรตอบภายในวันนี้", "ด่วน ลูกค้าเสียหายหรือใช้งานไม่ได้อยู่"]},
    "frustration": {"type": "score", "instructions": "ลูกค้าไม่พอใจแค่ไหน", "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]},
    "wants_refund": {"type": "noul", "instructions": "ลูกค้าขอเงินคืนหรือถามเรื่องการคืนเงินหรือไม่"},
    "wants_human": {"type": "noul", "instructions": "ลูกค้าต้องการคุยกับพนักงานที่เป็นคนจริงหรือไม่"},
    "has_order_ref": {"type": "noul", "instructions": "ข้อความระบุหมายเลขคำสั่งซื้อ ใบแจ้งหนี้ หรือเลขอ้างอิงหรือไม่"},
    "sentiment": {"type": "choice", "instructions": "อารมณ์โดยรวมของข้อความ",
                  "criteria": {"positive": "ชม พอใจ ดีใจ", "neutral": "เล่าเฉย ๆ ให้ข้อมูล", "negative": "บ่น ไม่พอใจ โกรธ", "question": "ถามคำถาม ขอข้อมูล"}},
}
ORDER = ["intent", "category", "department", "urgency", "frustration", "wants_refund", "wants_human", "has_order_ref", "sentiment"]
WISESIGHT = {0: "positive", 1: "neutral", 2: "negative", 3: "question"}


def with_other():
    """The run 5 form of the question set: intent gets the `other` option."""
    QUESTIONS["intent"]["criteria"]["other"] = OTHER_INTENT
    return QUESTIONS
