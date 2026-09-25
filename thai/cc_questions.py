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


# ---- debt collection: what a debtor's reply to a collection contact means (zero-shot for both models so far)
DEBT_QUESTIONS = {
    "intent": {"type": "choice", "instructions": "ลูกหนี้ต้องการอะไร หรือกำลังบอกอะไร", "criteria": {
        "promise_to_pay": "รับปากว่าจะจ่าย ระบุวัน/ช่องทางที่จะจ่าย",
        "request_extension": "ขอเลื่อนวันจ่าย ขอผ่อนผันไปก่อน ยังไม่มีเงินตอนนี้",
        "request_installment": "ขอแบ่งจ่าย ผ่อนเป็นงวด ขอลดยอด/ลดดอกเบี้ย ปรับโครงสร้างหนี้",
        "ask_balance": "ถามยอดคงเหลือ ดอกเบี้ย ค่าปรับ วิธีจ่าย เลขบัญชี",
        "already_paid": "แจ้งว่าจ่ายแล้ว ขอให้ตรวจสอบ/ยืนยันยอด ขอใบเสร็จ",
        "dispute_debt": "โต้แย้งว่ายอดไม่ถูก ไม่เคยกู้ ไม่ใช่หนี้ของตน ขอหลักฐาน",
        "wrong_person": "ไม่ใช่ลูกหนี้ โทรผิดคน เป็นญาติ/เพื่อน ไม่รู้จัก",
        "hardship": "เล่าปัญหา ตกงาน ป่วย รายได้ไม่พอ เพื่อขอความเห็นใจ",
        "complaint_harassment": "ร้องเรียน ถูกโทรบ่อย ข่มขู่ พูดไม่สุภาพ อ้างกฎหมายคุ้มครองลูกหนี้ จะแจ้งความ/ทนาย",
        "refuse_to_pay": "ปฏิเสธไม่จ่าย ไม่สนใจ ให้ฟ้องเลย",
        "callback_later": "ขอให้ติดต่อใหม่ภายหลัง ไม่สะดวกคุยตอนนี้ ให้ติดต่อคนอื่นแทน",
        "other": "ไม่เข้าข่ายข้อใดข้างต้น หรือไม่เกี่ยวกับหนี้",
    }},
    "willingness": {"type": "score", "instructions": "ลูกหนี้เต็มใจจะชำระแค่ไหน",
                    "criteria": ["ปฏิเสธหรือเลี่ยง", "ลังเล มีเงื่อนไข ขอเวลา", "ยินดีจ่าย ระบุแผนชัด"]},
    "frustration": {"type": "score", "instructions": "ลูกหนี้ไม่พอใจหรือโกรธแค่ไหน", "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]},
    "promise_to_pay": {"type": "noul", "instructions": "ลูกหนี้รับปากว่าจะชำระ (มีวันหรือแผนที่ชัดเจน) หรือไม่"},
    "financial_hardship": {"type": "noul", "instructions": "ลูกหนี้อ้างปัญหาทางการเงินหรือชีวิต (ตกงาน ป่วย รายได้ไม่พอ) หรือไม่"},
    "legal_or_complaint": {"type": "noul", "instructions": "ลูกหนี้อ้างกฎหมาย ขู่ร้องเรียน แจ้งความ หรือทนาย หรือไม่"},
    "disputes_debt": {"type": "noul", "instructions": "ลูกหนี้โต้แย้งว่าหนี้หรือยอดไม่ถูกต้อง หรือบอกว่าจ่ายแล้ว หรือไม่"},
    "next_action": {"type": "choice", "instructions": "เจ้าหน้าที่ควรทำอะไรต่อ", "criteria": {
        "send_payment_info": "ส่งยอด/ช่องทางชำระ/เลขบัญชี ให้ข้อมูล",
        "offer_plan": "เสนอแผนผ่อน เลื่อนนัด ปรับโครงสร้าง",
        "verify_payment": "ตรวจสอบยอดหรือการชำระในระบบ ส่งหลักฐาน",
        "schedule_followup": "นัดติดต่อใหม่ตามวันที่ลูกหนี้บอก",
        "escalate_supervisor": "ส่งต่อหัวหน้า/ฝ่ายร้องเรียน เพราะถูกร้องเรียนหรืออ้างกฎหมาย",
        "escalate_legal": "ส่งต่อฝ่ายกฎหมาย เพราะปฏิเสธจ่ายชัดเจน",
        "close_wrong_person": "ยุติ ผิดคน หรือไม่ใช่ลูกหนี้",
    }},
    "sentiment": {"type": "choice", "instructions": "อารมณ์โดยรวมของข้อความ",
                  "criteria": {"positive": "ให้ความร่วมมือ สุภาพ", "neutral": "เล่าเฉย ๆ ให้ข้อมูล", "negative": "บ่น ไม่พอใจ โกรธ", "question": "ถามคำถาม ขอข้อมูล"}},
}
DEBT_ORDER = list(DEBT_QUESTIONS)
QUESTION_SETS = {"ecom": lambda: (with_other(), ORDER), "debt": lambda: (DEBT_QUESTIONS, DEBT_ORDER)}


def get_questions(name="ecom"):
    return QUESTION_SETS[name]()
