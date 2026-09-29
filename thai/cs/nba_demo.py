"""Next best action demo (banking / telecom / insurance): the served call-center model (:8011) reads the message, the playbook
table turns its intent probabilities into action scores, the context rule adjusts for verification and repeat contacts.

    p(action) = sum of p(intent) over the intents whose table action, after the context rule, is that action

    python nba_demo.py "บัตรเครดิตหายเมื่อวาน ช่วยด้วยครับ"
    python nba_demo.py --business telecom --unverified --contacts 3 "เน็ตบ้านหลุดทุกคืน แจ้งไปสองรอบแล้ว"
"""
import argparse
import json
import sys
import time
import urllib.request
from collections import defaultdict

from cs_questions import SHARED, intent_question
from nba_actions import PLAYBOOKS, apply_context


def suggest(url: str, message: str, business: str = "banking", verified: bool = True, contacts: int = 1, k: int = 3) -> dict:
    qs = {"intent": intent_question(business), "urgency": SHARED["urgency"], "frustration": SHARED["frustration"]}
    body = json.dumps({"state": message, "questions": qs}, ensure_ascii=False).encode()
    t = time.perf_counter()
    with urllib.request.urlopen(urllib.request.Request(f"{url}/v1/systemone", body, {"content-type": "application/json"}), timeout=30) as r:
        ans = json.load(r)["answers"]
    ms = (time.perf_counter() - t) * 1000
    pb = PLAYBOOKS[business]
    score = defaultdict(float)
    for intent, p in ans["intent"]["probabilities"].items():
        score[apply_context(business, pb["intent_default"].get(intent, "out_of_scope"), verified, contacts)] += p
    top = sorted(score.items(), key=lambda kv: -kv[1])[:k]
    return {"actions": [(a, round(p, 3), pb["actions"][a]) for a, p in top],
            "intent": max(ans["intent"]["probabilities"].items(), key=lambda kv: kv[1]),
            "urgency": round(ans["urgency"]["score"], 2), "frustration": round(ans["frustration"]["score"], 2), "ms": round(ms)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="+")
    ap.add_argument("--url", default="http://172.18.72.145:8011")
    ap.add_argument("--business", default="banking", choices=sorted(PLAYBOOKS))
    ap.add_argument("--unverified", action="store_true")
    ap.add_argument("--contacts", type=int, default=1)
    args = ap.parse_args()
    for msg in args.message:
        s = suggest(args.url, msg, args.business, not args.unverified, args.contacts)
        print(f"\n{msg}\n  intent {s['intent'][0]} ({s['intent'][1]:.2f}), urgency {s['urgency']}/2, frustration {s['frustration']}/2, {s['ms']} ms")
        for i, (a, p, d) in enumerate(s["actions"], 1):
            print(f"  {i}. {a:22s} {p:.2f}  {d}")
        if s["frustration"] >= 1.5:
            print("  ! customer very upset: consider escalate_supervisor")


if __name__ == "__main__":
    sys.exit(main())
