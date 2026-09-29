"""Next best action (NBA) for the call center (banking / telecom / insurance): what the agent should do next, from the
customer's message plus two facts the agent's screen already knows (identity verified? how many times has the customer
contacted us about this?).

DRAFT playbooks: replace the actions and descriptions with the real call center's; labelling, the eval and the demo all
import this file.

The context policy is explicit (`apply_context`), so the gold answer for a message in any context follows from one human
label (the action for a verified, first-contact customer). The same policy is the rule baseline: intent (laya) ->
PLAYBOOKS[business]["intent_default"] -> apply_context.
"""

_COMMON = {
    "escalate_supervisor": "ขอโทษและส่งต่อหัวหน้างาน/ทีมเรื่องร้องเรียน (โกรธมาก ร้องเรียน ติดต่อซ้ำแต่ยังไม่ได้แก้)",
    "ask_clarification": "ข้อความยังไม่ชัดว่าต้องการอะไร: ถามข้อมูลเพิ่ม",
    "out_of_scope": "ไม่ใช่เรื่องของบริษัทนี้: แจ้งลูกค้าอย่างสุภาพและแนะนำหน่วยงานที่เกี่ยวข้อง",
}

BANKING = {
    "actions": {
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
        **_COMMON,
    },
    # actions on the customer's own account/card: an unverified caller must be verified first
    "needs_verification": {"block_card", "fraud_freeze", "open_dispute", "trace_transaction", "atm_case", "card_service", "card_troubleshoot",
                           "send_document", "update_details", "application_status", "debt_relief", "retention_offer", "process_refund"},
    # emergencies: act first, verify within the same call
    "urgent": {"block_card", "fraud_freeze"},
    # a repeated, still-unresolved contact about these becomes an escalation
    "escalate_if_repeated": {"open_dispute", "trace_transaction", "atm_case", "card_troubleshoot", "password_or_app_help", "kyc_help",
                             "process_refund", "application_status", "card_service", "send_document", "update_details"},
    "intent_default": {
        "activate_card": "card_service", "block_card": "block_card", "activate_card_international_usage": "card_service",
        "cancel_card": "retention_offer", "check_card_annual_fee": "give_information", "check_current_balance_on_card": "give_information",
        "apply_for_mortgage": "sales_referral", "cancel_mortgage": "retention_offer", "apply_for_loan": "sales_referral", "cancel_loan": "retention_offer",
        "check_mortgage_payments": "give_information", "check_loan_payments": "give_information", "make_transfer": "give_information",
        "cancel_transfer": "trace_transaction", "check_fees": "give_information", "check_recent_transactions": "send_document",
        "close_account": "retention_offer", "create_account": "sales_referral", "human_agent": "ask_clarification",
        "customer_service": "branch_or_atm_location", "recover_swallowed_card": "atm_case", "dispute_ATM_withdrawal": "atm_case",
        "find_branch": "branch_or_atm_location", "find_ATM": "branch_or_atm_location", "set_up_password": "password_or_app_help",
        "get_password": "password_or_app_help", "card_delivery": "card_service", "card_not_working": "card_troubleshoot",
        "card_replacement": "card_service", "apply_for_card": "sales_referral", "card_benefits_or_rewards": "give_information",
        "transfer_problem": "trace_transaction", "unrecognised_or_wrong_charge": "open_dispute", "refund_or_reversal": "process_refund",
        "top_up_or_deposit_problem": "trace_transaction", "foreign_currency_and_travel": "give_information",
        "identity_verification_kyc": "kyc_help", "account_suspended_or_fraud": "fraud_freeze", "app_or_login_problem": "password_or_app_help",
        "statement_or_document": "send_document", "change_personal_details": "update_details", "loan_or_debt_restructuring": "debt_relief",
        "deposit_or_interest_info": "give_information", "payment_or_bill": "trace_transaction", "complaint": "escalate_supervisor",
        "other": "out_of_scope",
    },
}

TELECOM = {
    "actions": {
        "verify_identity": "ยืนยันตัวตนผู้ใช้เบอร์ก่อน (ถามข้อมูลผู้จดทะเบียน/OTP) เพราะต้องเข้าถึงข้อมูลเบอร์หรือบัญชี",
        "suspend_sim": "ระงับเบอร์/ซิมทันที (ซิมหาย โทรศัพท์หาย ถูกขโมย) แล้วแนะนำการออกซิมใหม่",
        "troubleshoot_service": "แนะนำแก้ปัญหาเบื้องต้น (รีสตาร์ท ตั้งค่า APN เช็กซิม/เครื่อง) กรณีโทรไม่ได้ เน็ตใช้ไม่ได้ ใช้งานผิดปกติ",
        "network_ticket": "ตรวจสถานะเครือข่ายในพื้นที่ แจ้งเหตุขัดข้อง เปิดเคสเทคนิคหรือนัดช่าง (สัญญาณอ่อน เน็ตช้า เน็ตบ้านล่ม)",
        "billing_review": "ตรวจสอบบิล/ค่าบริการที่ลูกค้าโต้แย้ง อธิบายรายการ และเปิดเคสปรับยอดถ้าผิด",
        "compensation": "พิจารณาชดเชย ส่วนลด หรือคืนเงินจากปัญหาบริการตามเงื่อนไข",
        "account_info": "แจ้งข้อมูลเบอร์ของลูกค้า: ยอดใช้งาน เน็ต/นาทีคงเหลือ ยอดบิล สำเนาใบแจ้งหนี้ ค่าใช้จ่ายส่วนเกิน ตั้งเพดานค่าใช้จ่าย",
        "payment_help": "รับชำระ แนะนำช่องทางชำระ ตั้งหักอัตโนมัติ หรือตรวจยอดที่จ่ายแล้วแต่ยังไม่อัปเดต",
        "topup_case": "ตรวจรายการเติมเงิน เติมผิดเบอร์ ยอดไม่เข้า และเปิดเคสดึงเงินคืน",
        "plan_sales": "แนะนำ/สมัคร/เปลี่ยนแพ็กเกจ โปรโมชั่น เปิดเบอร์ใหม่ ติดตั้งเน็ตบ้าน",
        "retention_offer": "ลูกค้าจะยกเลิกบริการหรือย้ายค่าย: ถามเหตุผลและเสนอข้อเสนอก่อนดำเนินการ",
        "sim_service": "เปิดใช้/ลงทะเบียนซิม เปลี่ยนซิม eSIM เปิดเบอร์คืนหลังถูกระงับหรือหมดอายุ",
        "feature_service": "เปิด/ปิดบริการเสริม โรมมิ่ง โอนสาย บริการคอนเทนต์ ยกเลิก SMS หักเงิน",
        "device_service": "เรื่องเครื่อง/อุปกรณ์: ผ่อนเครื่อง ประกันเครื่อง ส่งซ่อม เปลี่ยนเราเตอร์หรือกล่อง",
        "update_details": "แก้ไขข้อมูลผู้ใช้ เปลี่ยนชื่อผู้จดทะเบียน โอนสิทธิ์เบอร์ ย้ายที่ติดตั้ง",
        "give_information": "ตอบข้อมูลทั่วไป พื้นที่ให้บริการ ค่าธรรมเนียม เงื่อนไข ช่องทางติดต่อ วิธีทำรายการ",
        **_COMMON,
    },
    "needs_verification": {"suspend_sim", "billing_review", "compensation", "account_info", "topup_case", "retention_offer", "sim_service",
                           "feature_service", "device_service", "update_details"},
    "urgent": {"suspend_sim"},
    "escalate_if_repeated": {"troubleshoot_service", "network_ticket", "billing_review", "compensation", "topup_case", "sim_service",
                             "feature_service", "device_service", "update_details", "payment_help"},
    "intent_default": {
        "dispute_invoice": "billing_review", "invoices": "account_info", "get_compensation": "compensation",
        "report_poor_signal_coverage": "network_ticket", "report_problem": "troubleshoot_service", "check_excess_data_charges": "account_info",
        "check_usage": "account_info", "set_usage_limits": "account_info", "customer_service": "give_information",
        "human_agent": "ask_clarification", "check_mobile_payments": "payment_help", "payment_methods": "payment_help", "pay": "payment_help",
        "schedule_payments": "payment_help", "activate_call_management_services": "feature_service",
        "deactivate_call_management_services": "feature_service", "activate_phone": "sim_service", "deactivate_phone": "suspend_sim",
        "activate_roaming": "feature_service", "check_signal_coverage": "give_information", "install_internet": "plan_sales",
        "cancel_plan": "retention_offer", "change_plan": "plan_sales", "change_provider": "retention_offer",
        "check_cancellation_fee": "give_information", "sign_up_for_plan": "plan_sales", "top_up_problem": "topup_case",
        "device_or_equipment": "device_service", "change_account_details": "update_details", "value_added_service": "feature_service",
        "complaint": "escalate_supervisor", "other": "out_of_scope",
    },
}

INSURANCE = {
    "actions": {
        "verify_identity": "ยืนยันตัวตนผู้เอาประกันก่อน (เลขกรมธรรม์/ข้อมูลส่วนตัว) เพราะต้องเข้าถึงข้อมูลกรมธรรม์",
        "claim_intake": "รับแจ้งเหตุ/แจ้งเคลมทันที บันทึกรายละเอียด ส่งพนักงานสำรวจ และแจ้งเอกสารที่ต้องใช้",
        "claim_status": "ตรวจสถานะเคลมหรือเงินสินไหม แจ้งความคืบหน้าและเอกสารที่ขาด",
        "settlement": "รับเรื่องตอบรับ ปฏิเสธ หรือต่อรองข้อเสนอค่าสินไหม และส่งต่อเจ้าหน้าที่สินไหม",
        "claim_appeal": "รับเรื่องอุทธรณ์เคลมที่ถูกปฏิเสธ อธิบายเหตุผลและขั้นตอน",
        "coverage_check": "ตรวจความคุ้มครองในกรมธรรม์ของลูกค้าว่าครอบคลุมกรณีนี้หรือไม่",
        "policy_change": "เปลี่ยน เพิ่ม หรือลดความคุ้มครอง แก้ข้อมูลส่วนตัวหรือผู้รับผลประโยชน์ในกรมธรรม์",
        "sales_quote": "แนะนำ/เปรียบเทียบแผนประกัน คำนวณเบี้ย ออกใบเสนอราคา และส่งต่อทีมขาย",
        "renewal": "ต่ออายุกรมธรรม์ แจ้งเบี้ยและเงื่อนไขปีถัดไป",
        "premium_payment": "รับชำระเบี้ย แนะนำช่องทางชำระ ตั้งหักอัตโนมัติ ตรวจประวัติการชำระ",
        "payment_issue": "ตรวจปัญหาการชำระเบี้ย หักซ้ำ ยอดผิด แล้วเปิดเคสแก้ไขหรือคืนเงิน",
        "retention_offer": "ลูกค้าจะยกเลิกหรือเวนคืนกรมธรรม์: ถามเหตุผล อธิบายผลกระทบและทางเลือกก่อนดำเนินการ",
        "send_policy_document": "ส่งกรมธรรม์ สำเนา เลขกรมธรรม์ หรือหนังสือรับรอง",
        "appointment": "นัดตรวจสภาพ นัดพบเจ้าหน้าที่หรือตัวแทน",
        "connect_agent": "ส่งต่อตัวแทน/นายหน้าหรือเจ้าหน้าที่ที่ดูแลกรมธรรม์",
        "give_information": "ตอบข้อมูลทั่วไป เงื่อนไข ค่าธรรมเนียม ขั้นตอน ช่องทางติดต่อ",
        **_COMMON,
    },
    "needs_verification": {"claim_intake", "claim_status", "settlement", "claim_appeal", "coverage_check", "policy_change", "renewal",
                           "payment_issue", "retention_offer", "send_policy_document"},
    "urgent": {"claim_intake"},
    "escalate_if_repeated": {"claim_status", "settlement", "claim_appeal", "payment_issue", "send_policy_document", "policy_change",
                             "coverage_check", "appointment", "renewal"},
    "intent_default": {
        "information_auto_insurance": "sales_quote", "accept_settlement": "settlement", "file_claim": "claim_intake",
        "negotiate_settlement": "settlement", "receive_payment": "claim_status", "reject_settlement": "settlement",
        "track_claim": "claim_status", "appeal_denied_insurance_claim": "claim_appeal", "dispute_invoice": "payment_issue",
        "file_complaint": "escalate_supervisor", "agent": "connect_agent", "customer_service": "give_information",
        "human_agent": "ask_clarification", "insurance_representative": "connect_agent", "change_coverage": "policy_change",
        "check_coverage": "coverage_check", "downgrade_coverage": "policy_change", "upgrade_coverage": "policy_change",
        "buy_insurance_policy": "sales_quote", "cancellation_fees": "give_information", "cancel_insurance_policy": "retention_offer",
        "compare_insurance_policies": "sales_quote", "general_information": "give_information", "information_health_insurance": "sales_quote",
        "information_home_insurance": "sales_quote", "report_incident": "claim_intake", "schedule_appointment": "appointment",
        "information_life_insurance": "sales_quote", "check_payments": "premium_payment", "payment_methods": "premium_payment",
        "pay": "premium_payment", "report_payment_issue": "payment_issue", "schedule_payments": "premium_payment",
        "information_pet_insurance": "sales_quote", "change_personal_details": "policy_change", "calculate_insurance_quote": "sales_quote",
        "check_rates": "sales_quote", "renew_insurance_policy": "renewal", "information_travel_insurance": "sales_quote",
        "policy_document_not_received": "send_policy_document", "other": "out_of_scope",
    },
}

PLAYBOOKS = {"banking": BANKING, "telecom": TELECOM, "insurance": INSURANCE}

VERIFIED = {True: "ยืนยันตัวตนแล้ว", False: "ยังไม่ได้ยืนยันตัวตน"}
HISTORY = {1: "ติดต่อเรื่องนี้เป็นครั้งแรก", 3: "ติดต่อเรื่องนี้เป็นครั้งที่ 3 แล้ว เรื่องยังไม่ได้รับการแก้ไข"}
CONTEXTS = [(v, h) for v in (True, False) for h in (1, 3)]


def apply_context(business: str, base: str, verified: bool, contacts: int) -> str:
    """The playbook rule: the action for a verified first contact (`base`) -> the action in this context."""
    pb = PLAYBOOKS[business]
    if contacts >= 3 and base in pb["escalate_if_repeated"]:
        return "escalate_supervisor"
    if not verified and base in pb["needs_verification"] and base not in pb["urgent"]:
        return "verify_identity"
    return base


def state_of(message: str, verified: bool, contacts: int) -> dict:
    return {"ข้อความลูกค้า": message, "สถานะการยืนยันตัวตน": VERIFIED[verified], "ประวัติการติดต่อ": HISTORY[contacts]}


def nba_question(business: str) -> dict:
    return {"type": "choice", "instructions": "พนักงานควรทำอะไรเป็นขั้นตอนถัดไป", "criteria": PLAYBOOKS[business]["actions"]}


def nba_base_question(business: str) -> dict:
    """What the teacher is asked (message only): it ignored the context fields in a smoke test (verify_identity top-1 on 18/40
    messages marked verified), so it judges the message for the base context and apply_context does the rest."""
    return {"type": "choice", "instructions": "ลูกค้ายืนยันตัวตนแล้วและเพิ่งติดต่อเรื่องนี้เป็นครั้งแรก พนักงานควรทำอะไรเป็นขั้นตอนถัดไป",
            "criteria": {k: v for k, v in PLAYBOOKS[business]["actions"].items() if k != "verify_identity"}}
