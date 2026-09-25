"""Customer-service question set for telecom / banking / insurance. Intents follow the Bitext taxonomies (the training text
comes from those sets) with Thai descriptions, which is what the models read. Two levels: `business` (which line of
business the message is about) and a per-business `intent`. The shared questions (department, urgency, frustration,
wants_human, wants_refund, has_reference, sentiment) are the same for all three.

Edit the descriptions here when the real call center's wording differs; both labelling and the review tools import this."""

INTENTS = {
    "telecom": {
        "dispute_invoice": "ค่าบริการ/ใบแจ้งหนี้ผิด มีรายการที่ไม่รู้จัก ขอโต้แย้งยอด",
        "invoices": "ขอดู/ขอสำเนาใบแจ้งหนี้ ถามยอดที่ต้องจ่าย",
        "get_compensation": "ขอชดเชย ขอส่วนลดหรือเงินคืนจากปัญหาบริการ",
        "report_poor_signal_coverage": "สัญญาณอ่อน สัญญาณหาย เน็ตช้า ในพื้นที่",
        "report_problem": "แจ้งปัญหาการใช้งานอื่น ๆ โทรไม่ออก เน็ตใช้ไม่ได้ ซิมมีปัญหา",
        "check_excess_data_charges": "ถามค่าบริการส่วนเกิน ค่าเน็ตเกินแพ็กเกจ",
        "check_usage": "ถามยอดใช้งาน เน็ตเหลือเท่าไหร่ นาทีโทรเหลือเท่าไหร่",
        "set_usage_limits": "ตั้งค่าจำกัดการใช้งาน ตั้งเพดานค่าใช้จ่าย",
        "customer_service": "ถามช่องทางติดต่อ เวลาทำการ เบอร์ศูนย์บริการ",
        "human_agent": "ขอคุยกับพนักงานที่เป็นคนจริง",
        "check_mobile_payments": "ถามเรื่องการจ่ายผ่านมือถือ/แอป ยอดที่จ่ายไปแล้ว",
        "payment_methods": "ถามว่ามีวิธีชำระเงินแบบไหนบ้าง",
        "pay": "ต้องการชำระค่าบริการตอนนี้",
        "schedule_payments": "ตั้งค่าจ่ายอัตโนมัติ ตัดบัตร นัดวันชำระ",
        "activate_call_management_services": "เปิดบริการโอนสาย สายเรียกซ้อน ฝากข้อความ ฯลฯ",
        "deactivate_call_management_services": "ปิดบริการโอนสาย สายเรียกซ้อน ฝากข้อความ ฯลฯ",
        "activate_phone": "เปิดใช้งานซิม/เบอร์ใหม่ เปิดเครื่อง",
        "deactivate_phone": "ระงับ/ปิดเบอร์ ระงับซิมชั่วคราว ซิมหาย",
        "activate_roaming": "เปิดโรมมิ่ง ใช้งานต่างประเทศ",
        "check_signal_coverage": "ถามว่าพื้นที่นี้มีสัญญาณไหม ครอบคลุมหรือไม่",
        "install_internet": "ขอติดตั้งอินเทอร์เน็ตบ้าน ไฟเบอร์ นัดช่าง",
        "cancel_plan": "ขอยกเลิกแพ็กเกจ ยกเลิกบริการ",
        "change_plan": "ขอเปลี่ยนแพ็กเกจ อัปเกรด/ดาวน์เกรด",
        "change_provider": "ย้ายค่าย ย้ายเบอร์ไปค่ายอื่น หรือย้ายเข้ามา",
        "check_cancellation_fee": "ถามค่าธรรมเนียมยกเลิกก่อนครบสัญญา",
        "sign_up_for_plan": "สมัครแพ็กเกจใหม่ เปิดเบอร์ใหม่",
    },
    "banking": {
        "activate_card": "เปิดใช้งานบัตรใหม่",
        "block_card": "อายัดบัตร บัตรหาย บัตรถูกขโมย",
        "activate_card_international_usage": "เปิดใช้บัตรต่างประเทศ",
        "cancel_card": "ยกเลิกบัตร ปิดบัตร",
        "check_card_annual_fee": "ถามค่าธรรมเนียมรายปีของบัตร ขอยกเว้น",
        "check_current_balance_on_card": "ถามยอดคงเหลือ/วงเงินในบัตร",
        "apply_for_mortgage": "ขอสินเชื่อบ้าน",
        "cancel_mortgage": "ยกเลิก/ปิดสินเชื่อบ้าน",
        "apply_for_loan": "ขอสินเชื่อส่วนบุคคล กู้เงิน",
        "cancel_loan": "ยกเลิก/ปิดสินเชื่อ โปะหนี้",
        "check_mortgage_payments": "ถามค่างวดบ้าน ยอดผ่อน วันครบกำหนด",
        "check_loan_payments": "ถามค่างวดสินเชื่อ ยอดผ่อน วันครบกำหนด",
        "make_transfer": "โอนเงิน ตั้งโอนล่วงหน้า",
        "cancel_transfer": "ยกเลิกรายการโอน โอนผิด ขอเรียกเงินคืน",
        "check_fees": "ถามค่าธรรมเนียม (โอน ถอน บัญชี)",
        "check_recent_transactions": "ขอดูรายการเดินบัญชี รายการล่าสุด",
        "close_account": "ปิดบัญชี",
        "create_account": "เปิดบัญชีใหม่",
        "human_agent": "ขอคุยกับพนักงานที่เป็นคนจริง",
        "customer_service": "ถามช่องทางติดต่อ เวลาทำการ สาขา/คอลเซ็นเตอร์",
        "recover_swallowed_card": "บัตรถูกตู้ ATM ยึด/กลืน",
        "dispute_ATM_withdrawal": "กดเงินไม่ออกแต่ยอดถูกหัก รายการ ATM ผิด",
        "find_branch": "หาสาขาใกล้เคียง เวลาเปิดสาขา",
        "find_ATM": "หาตู้ ATM ใกล้เคียง",
        "set_up_password": "ตั้ง/เปลี่ยนรหัสผ่าน PIN",
        "get_password": "ลืมรหัสผ่าน/PIN ขอกู้คืน",
    },
    "insurance": {
        "information_auto_insurance": "ถามข้อมูลประกันรถยนต์",
        "accept_settlement": "ตกลงรับข้อเสนอค่าสินไหม",
        "file_claim": "ยื่นเคลม แจ้งเคลมใหม่",
        "negotiate_settlement": "ต่อรองค่าสินไหม ขอเพิ่มวงเงินชดเชย",
        "receive_payment": "ถามว่าเงินเคลมจะได้เมื่อไหร่ ยังไม่ได้รับเงินสินไหม",
        "reject_settlement": "ปฏิเสธข้อเสนอค่าสินไหม",
        "track_claim": "ติดตามสถานะเคลม",
        "appeal_denied_insurance_claim": "อุทธรณ์เคลมที่ถูกปฏิเสธ",
        "dispute_invoice": "โต้แย้งใบแจ้งหนี้/ค่าเบี้ยที่ผิด",
        "file_complaint": "ร้องเรียนบริการหรือตัวแทน",
        "agent": "ขอติดต่อตัวแทน/นายหน้าประกัน",
        "customer_service": "ถามช่องทางติดต่อ เวลาทำการ",
        "human_agent": "ขอคุยกับพนักงานที่เป็นคนจริง",
        "insurance_representative": "ขอคุยกับเจ้าหน้าที่ประกันโดยเฉพาะ",
        "change_coverage": "ขอเปลี่ยนความคุ้มครอง",
        "check_coverage": "ถามว่ากรมธรรม์คุ้มครองอะไรบ้าง คุ้มครองกรณีนี้ไหม",
        "downgrade_coverage": "ขอลดความคุ้มครอง/ลดเบี้ย",
        "upgrade_coverage": "ขอเพิ่มความคุ้มครอง",
        "buy_insurance_policy": "ซื้อ/สมัครประกันใหม่",
        "cancellation_fees": "ถามค่าธรรมเนียมยกเลิกกรมธรรม์",
        "cancel_insurance_policy": "ยกเลิกกรมธรรม์ เวนคืน",
        "compare_insurance_policies": "ขอเปรียบเทียบแผนประกัน",
        "general_information": "ถามข้อมูลทั่วไปเกี่ยวกับบริษัทหรือประกัน",
        "information_health_insurance": "ถามข้อมูลประกันสุขภาพ",
        "information_home_insurance": "ถามข้อมูลประกันบ้าน/อัคคีภัย",
        "report_incident": "แจ้งเหตุ อุบัติเหตุ ความเสียหาย",
        "schedule_appointment": "นัดหมาย ตรวจสภาพ พบเจ้าหน้าที่",
        "information_life_insurance": "ถามข้อมูลประกันชีวิต",
        "check_payments": "ถามประวัติ/สถานะการชำระเบี้ย",
        "payment_methods": "ถามวิธีชำระเบี้ย",
        "pay": "ต้องการชำระเบี้ยตอนนี้",
        "report_payment_issue": "ชำระเบี้ยไม่ผ่าน ถูกหักซ้ำ ปัญหาการจ่าย",
        "schedule_payments": "ตั้งค่าจ่ายเบี้ยอัตโนมัติ นัดวันชำระ",
        "information_pet_insurance": "ถามข้อมูลประกันสัตว์เลี้ยง",
        "change_personal_details": "แก้ไขข้อมูลส่วนตัวในกรมธรรม์ ที่อยู่ ผู้รับผลประโยชน์",
        "calculate_insurance_quote": "ขอใบเสนอราคา คำนวณเบี้ย",
        "check_rates": "ถามอัตราเบี้ย ราคาแผน",
        "renew_insurance_policy": "ต่ออายุกรมธรรม์",
        "information_travel_insurance": "ถามข้อมูลประกันเดินทาง",
    },
}
OTHER = "ไม่เข้าข่ายข้อใดข้างต้น หรือไม่ใช่เรื่องติดต่อฝ่ายบริการลูกค้า"
BUSINESS_Q = {"type": "choice", "instructions": "ข้อความนี้เกี่ยวกับบริการของธุรกิจใด",
              "criteria": {"telecom": "ค่ายมือถือ อินเทอร์เน็ต แพ็กเกจ ซิม สัญญาณ", "banking": "ธนาคาร บัตร บัญชี โอนเงิน สินเชื่อ ATM",
                           "insurance": "ประกัน กรมธรรม์ เคลม ความคุ้มครอง เบี้ย", "other": "ไม่ใช่ทั้งสามอย่าง หรือไม่ใช่เรื่องบริการลูกค้า"}}
SHARED = {
    "department": {"type": "choice", "instructions": "ควรส่งเรื่องนี้ให้ทีมใดรับผิดชอบ",
                   "criteria": {"billing": "ค่าบริการ ใบแจ้งหนี้ การชำระเงิน เบี้ย ค่าธรรมเนียม", "technical": "ใช้งานไม่ได้ สัญญาณ แอป ระบบขัดข้อง ติดตั้ง",
                                "account": "บัญชี บัตร รหัสผ่าน ข้อมูลส่วนตัว กรมธรรม์", "sales": "สมัคร ซื้อ เปลี่ยนแพ็กเกจ/แผน ขอใบเสนอราคา",
                                "claims": "เคลม สินไหม อุบัติเหตุ ชดเชย", "support": "ร้องเรียน ขอคุยกับพนักงาน ถามข้อมูลทั่วไป"}},
    "urgency": {"type": "score", "instructions": "เรื่องนี้เร่งด่วนแค่ไหน",
                "criteria": ["ไม่รีบ ถามข้อมูลทั่วไป", "ควรตอบภายในวันนี้", "ด่วน ลูกค้าเสียหายหรือใช้งานไม่ได้อยู่ (บัตรหาย เน็ตล่ม อุบัติเหตุ)"]},
    "frustration": {"type": "score", "instructions": "ลูกค้าไม่พอใจแค่ไหน", "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]},
    "wants_human": {"type": "noul", "instructions": "ลูกค้าต้องการคุยกับพนักงานที่เป็นคนจริงหรือไม่"},
    "wants_refund": {"type": "noul", "instructions": "ลูกค้าขอเงินคืน ชดเชย หรือโต้แย้งยอดที่ถูกเรียกเก็บหรือไม่"},
    "has_reference": {"type": "noul", "instructions": "ข้อความระบุเลขอ้างอิง เช่น เลขบัญชี เลขกรมธรรม์ เลขเคลม เลขใบแจ้งหนี้ หรือเบอร์โทร หรือไม่"},
    "sentiment": {"type": "choice", "instructions": "อารมณ์โดยรวมของข้อความ",
                  "criteria": {"positive": "ชม พอใจ ดีใจ", "neutral": "เล่าเฉย ๆ ให้ข้อมูล", "negative": "บ่น ไม่พอใจ โกรธ", "question": "ถามคำถาม ขอข้อมูล"}},
}


def intent_question(business, with_other=True):
    crit = dict(INTENTS[business])
    if with_other:
        crit["other"] = OTHER
    return {"type": "choice", "instructions": "ลูกค้าต้องการอะไร", "criteria": crit}


def question_set(business, with_other=True):
    """The full set for one business: business + intent (that business's list) + shared questions."""
    return {"business": BUSINESS_Q, "intent": intent_question(business, with_other), **SHARED}


ORDER = ["business", "intent", "department", "urgency", "frustration", "wants_human", "wants_refund", "has_reference", "sentiment"]
