"""DRAFT: the fine intents of cs_questions.INTENTS merged into the groups a call center would actually route or handle
differently (two intents are merged when the agent does the same thing with them). For the call-center team to review:
move an intent to another group, split a group, rename. Nothing is retrained: the model still answers the fine intent
question and its probabilities are summed per group (eval_groups.py, group_probs below).
"""
from cs_questions import INTENTS

GROUPS = {
    "banking": {
        "card_lost": ("บัตรหาย/ถูกขโมย ต้องอายัด", ["block_card"]),
        "card_usage": ("บัตรใช้ไม่ได้ เปิดใช้บัตร ออกบัตรใหม่ ติดตามการส่งบัตร",
                       ["activate_card", "activate_card_international_usage", "card_delivery", "card_not_working", "card_replacement"]),
        "card_apply_info": ("สมัครบัตร สิทธิประโยชน์ ค่าธรรมเนียมรายปี วงเงิน",
                            ["apply_for_card", "card_benefits_or_rewards", "check_card_annual_fee", "check_current_balance_on_card"]),
        "atm_problem": ("ปัญหาที่ตู้ ATM: เงินไม่ออก บัตรถูกยึด", ["recover_swallowed_card", "dispute_ATM_withdrawal"]),
        "transfer_payment": ("โอน จ่ายบิล เติม/ฝากเงิน ทั้งถามวิธีและมีปัญหา",
                             ["make_transfer", "cancel_transfer", "transfer_problem", "top_up_or_deposit_problem", "payment_or_bill"]),
        "wrong_charge_refund": ("ถูกหักเงินผิด รายการที่ไม่ได้ทำ ขอเงินคืน", ["unrecognised_or_wrong_charge", "refund_or_reversal"]),
        "fraud_suspended": ("ถูกหลอก บัญชีถูกระงับ/อายัด เงินหาย", ["account_suspended_or_fraud"]),
        "app_login_identity": ("แอปเข้าไม่ได้ รหัสผ่าน ยืนยันตัวตน",
                               ["app_or_login_problem", "set_up_password", "get_password", "identity_verification_kyc"]),
        "loan_apply": ("ขอสินเชื่อ/สินเชื่อบ้าน", ["apply_for_loan", "apply_for_mortgage"]),
        "loan_repay_debt": ("ค่างวด ปิดหนี้ ผ่อนไม่ไหว ปรับโครงสร้างหนี้",
                            ["check_loan_payments", "check_mortgage_payments", "loan_or_debt_restructuring", "cancel_loan", "cancel_mortgage"]),
        "account_open_close": ("เปิดบัญชี ปิดบัญชี ยกเลิกบัตร", ["create_account", "close_account", "cancel_card"]),
        "account_info_documents": ("รายการเดินบัญชี เอกสาร แก้ไขข้อมูลส่วนตัว",
                                   ["check_recent_transactions", "statement_or_document", "change_personal_details"]),
        "fees_rates_products": ("ค่าธรรมเนียม ดอกเบี้ย เงินฝาก/ลงทุน อัตราแลกเปลี่ยน",
                                ["check_fees", "deposit_or_interest_info", "foreign_currency_and_travel"]),
        "branch_contact": ("สาขา ตู้ ATM ช่องทางติดต่อ ขอคุยกับพนักงาน", ["find_branch", "find_ATM", "customer_service", "human_agent"]),
        "complaint": ("ร้องเรียน", ["complaint"]),
    },
    "telecom": {
        "bill_dispute": ("ค่าบริการผิด ค่าบริการส่วนเกิน ขอชดเชย", ["dispute_invoice", "get_compensation", "check_excess_data_charges"]),
        "bill_usage_info": ("ถามยอดบิล ยอดใช้งาน ตั้งเพดาน", ["invoices", "check_usage", "set_usage_limits"]),
        "payment": ("ชำระค่าบริการ วิธีจ่าย ตั้งจ่ายอัตโนมัติ", ["check_mobile_payments", "payment_methods", "pay", "schedule_payments"]),
        "top_up": ("เติมเงินผิดเบอร์ เติมแล้วไม่เข้า", ["top_up_problem"]),
        "network_problem": ("สัญญาณ/เน็ตมีปัญหา ใช้งานไม่ได้ ถามพื้นที่สัญญาณ",
                            ["report_poor_signal_coverage", "report_problem", "check_signal_coverage"]),
        "plan_new_change": ("สมัคร/เปลี่ยนแพ็กเกจ ติดตั้งเน็ตบ้าน", ["change_plan", "sign_up_for_plan", "install_internet"]),
        "cancel_port_out": ("ยกเลิกบริการ ย้ายค่าย ค่าปรับ", ["cancel_plan", "change_provider", "check_cancellation_fee"]),
        "sim_number": ("เปิด/ระงับซิม เบอร์ถูกระงับ ซิมหาย เปลี่ยนชื่อ/ข้อมูลผู้ใช้",
                       ["activate_phone", "deactivate_phone", "change_account_details"]),
        "add_on_services": ("บริการเสริม โรมมิ่ง คอนเทนต์ที่หักผ่านเบอร์",
                            ["activate_call_management_services", "deactivate_call_management_services", "activate_roaming", "value_added_service"]),
        "device": ("เครื่อง/อุปกรณ์ เราเตอร์ ผ่อนเครื่อง", ["device_or_equipment"]),
        "contact_complaint": ("ช่องทางติดต่อ ขอคุยกับพนักงาน ร้องเรียน", ["customer_service", "human_agent", "complaint"]),
    },
    "insurance": {
        "claim_new": ("แจ้งเหตุ/ยื่นเคลมใหม่", ["file_claim", "report_incident"]),
        "claim_follow_up": ("ติดตามสถานะเคลม/เงินสินไหม", ["track_claim", "receive_payment"]),
        "claim_settlement": ("รับ/ปฏิเสธ/ต่อรองค่าสินไหม อุทธรณ์",
                             ["accept_settlement", "negotiate_settlement", "reject_settlement", "appeal_denied_insurance_claim"]),
        "coverage_question": ("ถามว่าคุ้มครองกรณีนี้ไหม", ["check_coverage"]),
        "policy_change": ("เปลี่ยนความคุ้มครอง แก้ข้อมูลในกรมธรรม์",
                          ["change_coverage", "downgrade_coverage", "upgrade_coverage", "change_personal_details"]),
        "buy_quote_info": ("สนใจซื้อ ขอราคา เปรียบเทียบแผน ถามข้อมูลประกันแต่ละประเภท",
                           ["buy_insurance_policy", "compare_insurance_policies", "calculate_insurance_quote", "check_rates",
                            "information_auto_insurance", "information_health_insurance", "information_home_insurance",
                            "information_life_insurance", "information_pet_insurance", "information_travel_insurance"]),
        "renewal": ("ต่ออายุกรมธรรม์", ["renew_insurance_policy"]),
        "cancel_policy": ("ยกเลิก/เวนคืนกรมธรรม์ ค่าธรรมเนียมยกเลิก", ["cancel_insurance_policy", "cancellation_fees"]),
        "premium_payment": ("ชำระเบี้ย วิธีจ่าย ประวัติการจ่าย", ["check_payments", "payment_methods", "pay", "schedule_payments"]),
        "payment_problem": ("จ่ายเบี้ยไม่ผ่าน หักซ้ำ ยอดผิด", ["report_payment_issue", "dispute_invoice"]),
        "policy_document": ("ยังไม่ได้กรมธรรม์ ขอสำเนา/เอกสาร", ["policy_document_not_received"]),
        "contact_agent": ("ช่องทางติดต่อ ตัวแทน เจ้าหน้าที่ นัดหมาย",
                          ["agent", "customer_service", "human_agent", "insurance_representative", "schedule_appointment"]),
        "general_info": ("ถามข้อมูลทั่วไปเกี่ยวกับบริษัทหรือประกัน", ["general_information"]),
        "complaint": ("ร้องเรียน", ["file_complaint"]),
    },
}

GROUP_OF = {biz: {i: g for g, (_, members) in gs.items() for i in members} for biz, gs in GROUPS.items()}
for _biz, _gs in GROUPS.items():  # every fine intent sits in exactly one group
    _members = [i for _, m in _gs.values() for i in m]
    assert sorted(_members) == sorted(INTENTS[_biz]), (_biz, set(INTENTS[_biz]) ^ set(_members))


def group_of(business: str, intent: str) -> str:
    return "other" if intent == "other" else GROUP_OF[business][intent]


def group_probs(business: str, intent_probs: dict) -> dict:
    """Fine-intent probabilities (as returned by the model) -> probability per group."""
    out = {}
    for intent, p in intent_probs.items():
        g = group_of(business, intent)
        out[g] = out.get(g, 0.0) + p
    return out
