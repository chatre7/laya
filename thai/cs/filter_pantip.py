"""Real Thai first-person questions for telecom / banking / insurance from Pantip topics (amitysolution/Pantip_QA_200000_20220220).
Keeps the topic text (`prompt`, one per topic_id) when it carries strong domain signals; writes per-business JSONL for review
sheets / distillation inputs.

    python filter_pantip.py --out /work/thai/data/domain
"""
import argparse
import collections
import json
import re
from pathlib import Path

from datasets import load_dataset

STRONG = {  # any one of these is enough
    "telecom": ["ais", "เอไอเอส", "ทรูมูฟ", "truemove", "true online", "ทรูออนไลน์", "dtac", "ดีแทค", "3bb", "nt broadband", "tot", "ค่ายมือถือ", "ย้ายค่าย",
                "โรมมิ่ง", "เน็ตบ้าน", "ไฟเบอร์", "เติมเงิน", "แพ็กเกจเน็ต", "แพ็คเกจเน็ต", "โปรเน็ต", "ซิมเติมเงิน", "ซิมรายเดือน", "สัญญาณมือถือ", "เน็ตหลุด", "เน็ตช้า"],
    "banking": ["ธนาคาร", "กสิกร", "kbank", "k plus", "kplus", "ไทยพาณิชย์", "scb easy", "กรุงไทย", "krungthai", "กรุงศรี", "krungsri", "ttb", "ทหารไทย", "ออมสิน",
                "ธกส", "ธอส", "บัตรเครดิต", "บัตรเดบิต", "พร้อมเพย์", "โอนเงินผิด", "ตู้ atm", "ตู้เอทีเอ็ม", "สินเชื่อ", "รีไฟแนนซ์", "แอปธนาคาร", "mobile banking",
                "บัญชีธนาคาร", "สมุดบัญชี", "ดอกเบี้ยเงินฝาก", "ผ่อนบ้าน", "กู้บ้าน", "กู้ซื้อรถ"],
    "insurance": ["ประกัน", "กรมธรรม์", "เคลมประกัน", "เบี้ยประกัน", "aia", "เมืองไทยประกัน", "ไทยประกันชีวิต", "fwd", "อลิอันซ์", "allianz", "วิริยะ", "ทิพยประกัน",
                  "กรุงเทพประกัน", "ประกันรถ", "ประกันสุขภาพ", "ประกันชีวิต", "ประกันชั้น 1", "พ.ร.บ.", "สินไหม", "ความคุ้มครอง"],
}
QUESTION_HINTS = ["ไหม", "มั้ย", "ยังไง", "อย่างไร", "ทำไม", "ได้ไหม", "หรือเปล่า", "?", "ช่วย", "ขอ", "อยาก", "ต้อง", "ใคร", "ที่ไหน", "เท่าไหร่", "กี่"]
EXCLUDE = {  # false friends per business
    "insurance": ["ประกันสังคม", "รับประกัน", "ประกันตัว", "ประกันภัยสังคม", "#", "ep"],
    "banking": ["#"],
    "telecom": ["#รักแท้", "ep"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/work/thai/data/domain")
    ap.add_argument("--max-len", type=int, default=600)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("amitysolution/Pantip_QA_200000_20220220", split="train")
    print("rows", len(ds), "columns", ds.column_names, flush=True)
    topics = {}
    for r in ds:
        t = (r.get("prompt") or "").strip()
        if t and r["topic_id"] not in topics:
            topics[r["topic_id"]] = t
    print("unique topics", len(topics), flush=True)
    hits = collections.defaultdict(list)
    for tid, t in topics.items():
        low = t.lower()
        if len(t) > args.max_len or len(t) < 15:
            continue
        for biz, kws in STRONG.items():
            if any(k in low for k in kws):
                if any(x in low for x in EXCLUDE.get(biz, [])):
                    break
                q = any(h in t for h in QUESTION_HINTS)
                hits[biz].append({"topic_id": tid, "text": re.sub(r"\s+", " ", t), "question_like": q})
                break
    for biz, rows in hits.items():
        with open(out / f"pantip_{biz}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        nq = sum(r["question_like"] for r in rows)
        print(f"\n== {biz}: {len(rows)} topics ({nq} question-like), mean {sum(len(r['text']) for r in rows) // max(1, len(rows))} chars")
        for r in rows[:8]:
            print("   -", r["text"][:130])


if __name__ == "__main__":
    main()
