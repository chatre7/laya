"""Next best action (NBA) for the banking call center: what the agent should do next, from the customer's message plus two
facts the agent's screen already knows (identity verified? how many times has the customer contacted us about this?).

A DRAFT catalogue: replace the actions and descriptions with the real call center's playbook; labelling, training, the eval
and the demo all import this file.

The context policy is explicit here (`apply_context`), so the gold answer for a message in any context follows from one
human label (the action for a verified, first-contact customer). The same policy is the rule baseline: intent (laya) ->
INTENT_DEFAULT -> apply_context.
"""

ACTIONS = {
    "verify_identity": "ยืนยันตัวตนลูกค้าก่อน (ถามข้อมูลส่วนตัว/OTP) เพราะต้องเข้าถึงบัญชีหรือบัตร",
    "block_card": "อายัดบัตร/ระงับการใช้บัตรทันที (บัตรหาย ถูกขโมย ข้อมูลบัตรรั่ว)",
    "fraud_freeze": "ระงับบัญชีชั่วคราวและส่งต่อทีมป้องกันทุจริต (ถูกหลอกโอน มิจฉาชีพ บัญชีถูกแฮ็ก เงินหายจากบัญชี)",
    "open_dispute": "เปิดเคสโต้แย้งรายการ (รายการที่ไม่ได้ทำ หักซ้ำ หักเกิน) และแจ้งระยะเวลาตรวจสอบ",
    "trace_transaction": "ตรวจสอบสถานะรายการโอน/ฝาก/เติมเงิน/ชำระบิล และเปิดเคสติดตามถ้ายอดยังไม่เข้า",
    "atm_case": "เปิดเคสตรวจสอบตู้ ATM (กดเงินไม่ออกแต่ถูกหัก บัตรถูกตู้ยึด ตู้ฝากกินเงิน)",
    "card_service": "จัดการบัตร: เปิดใช้งาน ออกบัตรใหม่แทนบัตรหมดอายุ/ชำรุด ตรวจสถานะการจัดส่งบัตร เปิดใช้ต่างประเทศ",
    "card_troubleshoot": "ตรวจสาเหตุที่บัตรใช้ไม่ได้/ถูกปฏิเสธ แล้วแก้ไขหรือแนะนำวิธีแก้",
    "password_or_app_help": "แนะนำขั้นตอนรีเซ็ตรหัส/ปลดล็อก/ลงทะเบียนแอปใหม่ด้วยตนเอง หรือแจ้งสถานะระบบถ้าระบบล่ม",
    "kyc_help": "แนะนำขั้นตอนยืนยันตัวตน/ส่งเอกสาร KYC และสาเหตุที่ยืนยันไม่ผ่าน",
    "give_information": "ตอบข้อมูลผลิตภัณฑ์ ค่าธรรมเนียม ดอกเบี้ย อัตราแลกเปลี่ยน เงื่อนไข หรือวิธีทำรายการ",
    "send_document": "ส่งเอกสาร statement หนังสือรับรอง ใบเสร็จ ทางช่องทางที่ลูกค้าสะดวก",
    "update_details": "แก้ไขข้อมูลส่วนตัว ที่อยู่ เบอร์โทร อีเมล ชื่อ ให้ลูกค้า",
    "sales_referral": "แนะนำผลิตภัณฑ์ที่ตรงความต้องการ (บัตร สินเชื่อ บัญชี เงินฝาก) และส่งต่อทีมขายหรือส่งลิงก์สมัคร",
    "application_status": "ตรวจสถานะใบสมัครบัตร/สินเชื่อ แจ้งผลหรือเอกสารที่ขาด",
    "debt_relief": "เสนอทางเลือกผ่อนผัน/ปรับโครงสร้างหนี้ และส่งต่อทีมบริหารหนี้",
    "retention_offer": "ลูกค้าจะปิดบัญชี/ยกเลิกบัตร/ยกเลิกสินเชื่อ: ถามเหตุผลและเสนอทางเลือกก่อนดำเนินการ",
    "process_refund": "ดำเนินการคืนเงิน/คืนค่าธรรมเนียม/ยกเลิกรายการ หรือแจ้งสถานะเงินคืน",
    "branch_or_atm_location": "แจ้งสาขาหรือตู้ ATM ใกล้เคียง เวลาทำการ หรือเรื่องที่ต้องไปทำที่สาขา",
    "escalate_supervisor": "ขอโทษและส่งต่อหัวหน้างาน/ทีมเรื่องร้องเรียน (โกรธมาก ร้องเรียน ติดต่อซ้ำแต่ยังไม่ได้แก้)",
    "ask_clarification": "ข้อความยังไม่ชัดว่าต้องการอะไร: ถามข้อมูลเพิ่ม",
    "out_of_scope": "ไม่ใช่เรื่องของธนาคาร: แจ้งลูกค้าอย่างสุภาพและแนะนำหน่วยงานที่เกี่ยวข้อง",
}

# Actions that touch the customer's own account/card: an unverified caller must be verified first.
NEEDS_VERIFICATION = {"block_card", "fraud_freeze", "open_dispute", "trace_transaction", "atm_case", "card_service", "card_troubleshoot",
                      "send_document", "update_details", "application_status", "debt_relief", "retention_offer", "process_refund"}
# Emergencies: act first, verify within the same call (blocking a stolen card must not wait).
URGENT = {"block_card", "fraud_freeze"}
# Actions that a repeated, still-unresolved contact turns into an escalation.
ESCALATE_IF_REPEATED = {"open_dispute", "trace_transaction", "atm_case", "card_troubleshoot", "password_or_app_help", "kyc_help",
                        "process_refund", "application_status", "card_service", "send_document", "update_details"}

VERIFIED = {True: "ยืนยันตัวตนแล้ว", False: "ยังไม่ได้ยืนยันตัวตน"}
HISTORY = {1: "ติดต่อเรื่องนี้เป็นครั้งแรก", 3: "ติดต่อเรื่องนี้เป็นครั้งที่ 3 แล้ว เรื่องยังไม่ได้รับการแก้ไข"}
CONTEXTS = [(v, h) for v in (True, False) for h in (1, 3)]


def apply_context(base: str, verified: bool, contacts: int) -> str:
    """The playbook rule: the action for a verified first contact (`base`) -> the action in this context."""
    if contacts >= 3 and base in ESCALATE_IF_REPEATED:
        return "escalate_supervisor"
    if not verified and base in NEEDS_VERIFICATION and base not in URGENT:
        return "verify_identity"
    return base


def state_of(message: str, verified: bool, contacts: int) -> dict:
    return {"ข้อความลูกค้า": message, "สถานะการยืนยันตัวตน": VERIFIED[verified], "ประวัติการติดต่อ": HISTORY[contacts]}


NBA_Q = {"type": "choice", "instructions": "พนักงานควรทำอะไรเป็นขั้นตอนถัดไป", "criteria": ACTIONS}

# Rule baseline: the banking intent (cs_questions.INTENTS["banking"]) -> the usual action for a verified first contact.
INTENT_DEFAULT = {
    "activate_card": "card_service", "block_card": "block_card", "activate_card_international_usage": "card_service",
    "cancel_card": "retention_offer", "check_card_annual_fee": "give_information", "check_current_balance_on_card": "give_information",
    "apply_for_mortgage": "sales_referral", "cancel_mortgage": "retention_offer", "apply_for_loan": "sales_referral", "cancel_loan": "retention_offer",
    "check_mortgage_payments": "give_information", "check_loan_payments": "give_information", "make_transfer": "give_information",
    "cancel_transfer": "trace_transaction", "check_fees": "give_information", "check_recent_transactions": "send_document",
    "close_account": "retention_offer", "create_account": "sales_referral", "human_agent": "ask_clarification", "customer_service": "branch_or_atm_location",
    "recover_swallowed_card": "atm_case", "dispute_ATM_withdrawal": "atm_case", "find_branch": "branch_or_atm_location", "find_ATM": "branch_or_atm_location",
    "set_up_password": "password_or_app_help", "get_password": "password_or_app_help", "card_delivery": "card_service", "card_not_working": "card_troubleshoot",
    "card_replacement": "card_service", "apply_for_card": "sales_referral", "card_benefits_or_rewards": "give_information", "transfer_problem": "trace_transaction",
    "unrecognised_or_wrong_charge": "open_dispute", "refund_or_reversal": "process_refund", "top_up_or_deposit_problem": "trace_transaction",
    "foreign_currency_and_travel": "give_information", "identity_verification_kyc": "kyc_help", "account_suspended_or_fraud": "fraud_freeze",
    "app_or_login_problem": "password_or_app_help", "statement_or_document": "send_document", "change_personal_details": "update_details",
    "loan_or_debt_restructuring": "debt_relief", "deposit_or_interest_info": "give_information", "payment_or_bill": "trace_transaction",
    "complaint": "escalate_supervisor", "other": "out_of_scope",
}
