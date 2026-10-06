"""Second hand-check sample for the new questions. The first (sample_new_check.py) was drawn by the teacher's answers and
held 5 / 1 / 6 / 14 hand "yes" - too few to judge a teacher on. This one draws from the whole pool by wording: for each
question, texts that contain a word a real "yes" usually contains (leave, sue, mother, called ...). Most of them are still
"no" (cancel a package, "my mother told me"), which is the hard part.

    python sample_new_check2.py     # -> data_domain/new_check_sample2.jsonl, data_domain/_print_new_check2.txt
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from label_new_llm import pool  # noqa: E402

CUES = {
    "churn_threat": r"เลิกใช้|ย้ายค่าย|ย้ายไปใช้|ย้ายไป|ปิดบัญชี|ลบแอ[ปพ]|ถอนการติดตั้ง|ยกเลิก(บริการ|บัตร|ประกัน|กรมธรรม์|เบอร์|สัญญา|เน็ต)|ไม่ต่อ(อายุ|สัญญา|ประกัน)"
                    r"|เปลี่ยนไปใช้|เปลี่ยนค่าย|เปลี่ยนธนาคาร|ไม่ใช้แล้ว|เลิกเป็นลูกค้า|เจ้าอื่น|ค่ายอื่น|ธนาคารอื่น",
    "external_threat": r"สคบ|กสทช|แบงก์ชาติ|ธปท|คปภ|ฟ้อง|แจ้งความ|ร้องเรียน|ทนาย|ขึ้นศาล|ออกสื่อ|ลงโซเชียล|ประจาน|แฉ|ดำรงธรรม|คุ้มครองผู้บริโภค|ตำรวจ",
    "third_party": r"ของแม่|ของพ่อ|คุณแม่|คุณพ่อ|แม่ผม|พ่อผม|แม่เรา|พ่อเรา|ลูกชาย|ลูกสาว|ญาติ|สามี|ภรรยา|ของแฟน|ยาย|คุณย่า|คุณปู่|เสียชีวิต|ผู้สูงอายุ|มอบอำนาจ|ทำแทน|ติดต่อแทน",
    "contact_effort": r"โทรไป|โทรหา|โทรแจ้ง|โทรถาม|ติดต่อไป|ติดต่อแล้ว|ติดต่อไม่ได้|แจ้งไป|แจ้งเรื่อง|แจ้งแล้ว|ไปสาขา|ไปที่สาขา|ไปธนาคาร|ไปศูนย์|คอลเซ็นเตอร์|คอลเซนเตอร์"
                      r"|call ?cent|หลายครั้ง|หลายรอบ|สามรอบ|สองรอบ|รอสาย|แชท(ไป|ถาม)|ทักไป",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--have", default="/work/thai/data_domain/new_check_sample.jsonl")
    ap.add_argument("--out", default="/work/thai/data_domain/new_check_sample2.jsonl")
    ap.add_argument("--print", dest="print_file", default="/work/thai/data_domain/_print_new_check2.txt")
    ap.add_argument("--per-question", type=int, default=45)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    have = {json.loads(l)["id"] for l in open(args.have, encoding="utf-8")}
    rows = [r for r in pool() if r["id"] not in have]
    rng.shuffle(rows)
    picked = {}
    for q, pat in CUES.items():
        rx = re.compile(pat, re.I)
        hits = [r for r in rows if rx.search(r["text"])]
        print(q, "texts with a cue word:", len(hits), "of", len(rows))
        n = 0
        for r in hits:
            if n >= args.per_question:
                break
            if r["id"] not in picked:
                picked[r["id"]] = r
                n += 1
    sample = list(picked.values())
    rng.shuffle(sample)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps({**r, "text": r["text"][:1500]}, ensure_ascii=False) + "\n")
    with open(args.print_file, "w", encoding="utf-8") as f:
        for i, r in enumerate(sample):
            f.write(f"{i}|{r['id']}|{r['text'][:520]}\n")
    print(len(sample), "texts ->", args.out)


if __name__ == "__main__":
    main()
