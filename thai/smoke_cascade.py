"""Smoke test for the cascade server: contract, routing, a 60-option question, and latency at 8 concurrent callers.

    python thai/smoke_cascade.py [--url http://172.18.72.145:8011]
"""
import argparse
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TICKET = {
    "state": "โดนหักเงินซ้ำสองครั้งเมื่อวานนี้ ขอเงินคืนด่วนนะครับ",
    "questions": {
        "department": {"type": "choice", "instructions": "ทีมใดควรรับผิดชอบ",
                       "criteria": {"billing": "ค่าบริการ ใบแจ้งหนี้", "technical": "ระบบใช้งานไม่ได้", "sales": "สมัคร เปลี่ยนแพ็กเกจ"}},
        "frustration": {"type": "score", "instructions": "ลูกค้าไม่พอใจแค่ไหน", "criteria": ["ใจเย็น", "หงุดหงิดแต่สุภาพ", "โกรธมาก"]},
        "refund": {"type": "noul", "instructions": "ลูกค้าขอเงินคืนอย่างชัดเจนหรือไม่"},
    },
}
INTENTS = ["alarm_set", "alarm_query", "alarm_remove", "audio_volume_mute", "audio_volume_up", "audio_volume_down", "calendar_set",
           "calendar_query", "calendar_remove", "cooking_recipe", "datetime_query", "datetime_convert", "email_query", "email_sendemail",
           "email_addcontact", "general_greet", "general_joke", "general_quirky", "iot_hue_lightoff", "iot_hue_lighton", "iot_hue_lightdim",
           "iot_cleaning", "iot_coffee", "iot_wemo_on", "iot_wemo_off", "lists_createoradd", "lists_query", "lists_remove", "music_likeness",
           "music_query", "music_settings", "news_query", "play_audiobook", "play_game", "play_music", "play_podcasts", "play_radio",
           "qa_currency", "qa_definition", "qa_factoid", "qa_maths", "qa_stock", "recommendation_events", "recommendation_locations",
           "recommendation_movies", "social_post", "social_query", "takeaway_order", "takeaway_query", "transport_query", "transport_taxi",
           "transport_ticket", "transport_traffic", "weather_query", "iot_hue_lightchange", "iot_hue_lightup", "music_dislikeness",
           "email_querycontact", "audio_volume_other", "cooking_query"]
WIDE = {"state": "ตั้งปลุกตอนหกโมงเช้าพรุ่งนี้ให้หน่อย",
        "questions": {"intent": {"type": "choice", "instructions": "ผู้ใช้ต้องการอะไร", "criteria": {k: None for k in INTENTS}}}}
VAGUE = {"state": "อืม", "questions": {"topic": {"type": "choice", "instructions": "ข้อความนี้เกี่ยวกับอะไร",
                                                  "criteria": {"การเมือง": None, "กีฬา": None, "บันเทิง": None, "เศรษฐกิจ": None}}}}


def post(url, body):
    req = urllib.request.Request(url + "/v1/systemone", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    t = time.perf_counter()
    res = json.load(urllib.request.urlopen(req, timeout=120))
    return res, (time.perf_counter() - t) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://172.18.72.145:8011")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    print("healthz:", json.load(urllib.request.urlopen(url + "/healthz", timeout=10)))
    for name, body in [("ticket", TICKET), ("wide-intent (60 options)", WIDE), ("vague", VAGUE)]:
        res, ms = post(url, body)
        c = res["usage"]["cascade"]
        print(f"\n== {name}: {ms:.0f} ms, student {c['student_ms']} ms, teacher {c['teacher_ms']} ms, to teacher: {c['reasons'] or '-'}")
        for qid, a in res["answers"].items():
            key = a["type"]
            conf = max(a["probabilities"].values()) if "probabilities" in a else max(a["noul"], 1 - a["noul"])
            extra = "  abstain=%.2f" % a["abstain"] if a.get("abstain") is not None else ""
            print(f"   {qid:12s} {a['type']:6s} -> {a[key]!s:14s} max p={conf:.2f}{extra}")
    n, workers = 32, 8
    with ThreadPoolExecutor(workers) as ex:
        t = time.perf_counter()
        lat = list(ex.map(lambda i: post(url, TICKET)[1], range(n)))
        wall = time.perf_counter() - t
    lat.sort()
    print(f"\n== {n} ticket requests at {workers} concurrent: mean {sum(lat)/n:.0f} ms, p50 {lat[n//2]:.0f}, p95 {lat[int(n*0.95)]:.0f}, throughput {n/wall:.1f} req/s")
    print("stats:", json.load(urllib.request.urlopen(url + "/stats", timeout=10)))


if __name__ == "__main__":
    main()
