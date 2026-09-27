"""Re-label the `other` rows of the first Pantip review samples (labels_<biz>.json, the real eval set) where a run 8 intent now
fits (device_or_equipment, value_added_service, app_or_login_problem, loan_or_debt_restructuring, ...). One-off; keeps the file
format. Run once, then rebuild real_cs_eval.jsonl with build_real_eval.py."""
import json
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data_domain")
NEW = {
    "telecom": {"28": ["device_or_equipment", "technical", 0], "108": ["value_added_service", "support", 0], "119": ["value_added_service", "account", 0]},
    "banking": {
        "0": ["statement_or_document", "account", 0], "2": ["loan_or_debt_restructuring", "billing", 1], "7": ["change_personal_details", "account", 0],
        "8": ["deposit_or_interest_info", "sales", 0], "10": ["app_or_login_problem", "technical", 0], "21": ["payment_or_bill", "billing", 0],
        "22": ["apply_for_card", "sales", 0], "31": ["cancel_mortgage", "account", 0], "33": ["deposit_or_interest_info", "support", 0],
        "35": ["top_up_or_deposit_problem", "account", 0], "37": ["change_personal_details", "account", 0], "38": ["apply_for_card", "sales", 0],
        "39": ["check_current_balance_on_card", "account", 0], "43": ["account_suspended_or_fraud", "support", 1], "46": ["deposit_or_interest_info", "sales", 0],
        "47": ["change_personal_details", "account", 0], "50": ["customer_service", "support", 0], "56": ["statement_or_document", "account", 0],
        "59": ["deposit_or_interest_info", "account", 0], "67": ["identity_verification_kyc", "technical", 0], "69": ["refund_or_reversal", "billing", 1],
        "82": ["foreign_currency_and_travel", "technical", 0], "86": ["account_suspended_or_fraud", "account", 2], "88": ["apply_for_card", "sales", 0],
        "89": ["app_or_login_problem", "technical", 0], "91": ["payment_or_bill", "billing", 0], "92": ["apply_for_mortgage", "billing", 0],
        "93": ["apply_for_card", "sales", 0], "95": ["apply_for_card", "sales", 0], "101": ["check_current_balance_on_card", "account", 0],
        "102": ["app_or_login_problem", "technical", 0], "111": ["deposit_or_interest_info", "sales", 0], "116": ["payment_or_bill", "technical", 1],
        "117": ["app_or_login_problem", "technical", 1], "118": ["statement_or_document", "account", 0],
    },
    "insurance": {"115": ["check_coverage", "claims", 0]},
}
for biz, upd in NEW.items():
    p = os.path.join(D, f"labels_{biz}.json")
    d = json.load(open(p, encoding="utf-8"))
    n = 0
    for k, v in upd.items():
        assert d["labels"][k][0] == "other", (biz, k, d["labels"][k])
        d["labels"][k] = v
        n += 1
    d["note"] = d.get("note", "") + " | run 8: former `other` rows re-labelled with the extended intent lists (fix_labels_run8.py)"
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False)
    print(biz, "updated", n, "rows; other now", sum(1 for v in d["labels"].values() if v[0] == "other"), "/", len(d["labels"]))
