"""Probe out-of-scope detection on the cascade endpoint: the 27 intents + `other` (run 5 was trained with it)."""
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from check_rewrites import INTENTS  # noqa: E402

URL = sys.argv[1] if len(sys.argv) > 1 else "http://172.18.72.145:8011"
crit = dict(INTENTS)
crit["other"] = "ไม่เข้าข่ายข้อใดข้างต้น หรือไม่ใช่เรื่องติดต่อฝ่ายบริการลูกค้า"
for text in ["วันนี้อากาศดีจัง ไปเที่ยวทะเลกันไหม", "เพื่อไทยแถลงนโยบายใหม่เรื่องค่าแรงขั้นต่ำ", "ขอยกเลิกออเดอร์ 48213 ครับ สั่งผิดรุ่น",
             "โดนหักเงินซ้ำสองครั้ง ขอเงินคืนด่วน", "อืม", "ขอคุยกับพนักงานได้ไหม บอทตอบไม่ตรง"]:
    body = {"state": text, "questions": {"intent": {"type": "choice", "instructions": "ลูกค้าต้องการอะไร", "criteria": crit}}}
    req = urllib.request.Request(URL + "/v1/systemone", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    a = json.load(urllib.request.urlopen(req, timeout=60))
    i, c = a["answers"]["intent"], a["usage"]["cascade"]
    print(f"  {text[:34]:36s} -> {i['choice']:26s} p={max(i['probabilities'].values()):.2f}  to teacher: {c['reasons'] or '-'}")
