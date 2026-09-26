"""Probe the cascade endpoint with the four-business question set: business + that business's intent (+ other) + department."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cs_questions import BUSINESS_Q, SHARED, intent_question  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://172.18.72.145:8011"
CASES = [
    ("telecom", "เน็ตบ้านหลุดบ่อยมากตั้งแต่เมื่อวาน รีสตาร์ทแล้วก็ยังเป็น ช่วยส่งช่างมาดูหน่อย"),
    ("telecom", "สนใจเปลี่ยนเป็นแพ็กเกจ 5G ที่โฆษณาอยู่ ต้องทำยังไงบ้างคะ"),
    ("telecom", "จะไปญี่ปุ่นอาทิตย์หน้า เปิดโรมมิ่งยังไงครับ"),
    ("banking", "บัตรหายค่ะ ขออายัดด่วน"),
    ("banking", "โอนเงินผิดบัญชีไปเมื่อกี้ ขอเรียกเงินคืนได้ไหม"),
    ("banking", "อยากขอสินเชื่อบ้าน ต้องใช้เอกสารอะไรบ้าง"),
    ("insurance", "รถชนเมื่อเช้า จะแจ้งเคลมยังไงครับ กรมธรรม์ P-30291"),
    ("insurance", "เคลมที่ยื่นไปเดือนที่แล้วถึงไหนแล้วคะ ยังไม่ได้เงินเลย"),
    ("insurance", "อยากลดเบี้ยลง ปรับความคุ้มครองได้ไหม"),
    ("ecommerce", "ของยังไม่ถึงเลย สั่งไปตั้งแต่อาทิตย์ที่แล้ว เลขออเดอร์ 48213"),
    ("ecommerce", "ส่งมาแล้วแต่กล่องบุบ ของข้างในเสียหาย ขอเปลี่ยนหรือคืนเงิน"),
    ("telecom", "วันนี้อากาศดีจัง ไปเที่ยวทะเลกันไหม"),
    ("banking", "อืม"),
]
for biz, text in CASES:
    qs = {"business": BUSINESS_Q, "intent": intent_question(biz), "department": SHARED["department"]}
    body = {"state": text, "questions": qs}
    req = urllib.request.Request(URL + "/v1/systemone", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    a = json.load(urllib.request.urlopen(req, timeout=60))
    b, i, d, c = a["answers"]["business"], a["answers"]["intent"], a["answers"]["department"], a["usage"]["cascade"]
    print(f"  [{biz:9s}] {text[:34]:36s} -> business {b['choice']:9s} p={max(b['probabilities'].values()):.2f} | intent {i['choice']:30s} p={max(i['probabilities'].values()):.2f} | dept {d['choice']:9s} | teacher: {c['reasons'] or '-'}")
