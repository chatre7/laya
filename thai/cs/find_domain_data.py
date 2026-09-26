"""Look for Thai telecom / banking / insurance customer text: (1) Hugging Face datasets by keyword, (2) real Thai posts in
wisesight_sentiment (train + validation + test) that mention telcos, banks or insurers, written out per business for the
review sheet / distillation.

    python find_domain_data.py --out D:/cmtn-project/laya/thai/data_domain
"""
import argparse
import collections
import json
import os
import re

from huggingface_hub import HfApi

KEYWORDS = {
    "telecom": ["ais", "เอไอเอส", "true", "ทรู", "dtac", "ดีแทค", "nt ", "tot", "cat ", "3bb", "ค่ายมือถือ", "ซิม", "แพ็กเกจ", "แพ็คเกจ", "โปรเน็ต", "เน็ตบ้าน",
                "ไฟเบอร์", "สัญญาณ", "โรมมิ่ง", "เติมเงิน", "รายเดือน", "ย้ายค่าย", "5g", "4g", "wifi", "เน็ตช้า", "เน็ตหลุด"],
    "banking": ["ธนาคาร", "กสิกร", "kbank", "ไทยพาณิชย์", "scb", "กรุงเทพ", "bbl", "กรุงไทย", "ktb", "กรุงศรี", "ttb", "ทหารไทย", "ออมสิน", "ธกส", "บัตรเครดิต",
                "บัตรเดบิต", "โอนเงิน", "พร้อมเพย์", "atm", "สินเชื่อ", "กู้", "ดอกเบี้ย", "บัญชี", "แอปธนาคาร", "mobile banking", "k plus", "scb easy"],
    "insurance": ["ประกัน", "กรมธรรม์", "เคลม", "เบี้ย", "aia", "เมืองไทยประกัน", "ไทยประกันชีวิต", "fwd", "allianz", "อลิอันซ์", "วิริยะ", "ทิพยประกัน", "ประกันรถ",
                  "ประกันสุขภาพ", "ประกันชีวิต", "พรบ", "คุ้มครอง", "สินไหม"],
}
QUERIES = ["thai telecom", "thai telco", "thai bank", "thai banking", "thai insurance", "thai finance customer", "thai complaint", "thai customer service",
           "thai faq", "thai chatbot bank", "ประกัน", "ธนาคาร", "thai call center", "thai conversation customer", "thai support ticket", "thai review bank"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-hf", action="store_true", help="skip the Hugging Face search")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if not args.no_hf:
        api = HfApi()
        print("== Hugging Face datasets")
        seen = {}
        for q in QUERIES:
            for d in api.list_datasets(search=q, limit=40):
                seen.setdefault(d.id, d)
        rows = []
        for d in seen.values():
            tags = " ".join(d.tags or []).lower()
            idl = d.id.lower()
            if not ("language:th" in tags or "thai" in idl or "th" in idl.split("-") or any(k in idl for k in ("thai", "_th", "-th"))):
                continue
            rows.append((d.downloads or 0, d.id, [t for t in (d.tags or []) if t.startswith(("license:", "size_categories:"))]))
        for dl, i, t in sorted(rows, reverse=True)[:40]:
            print(f"  {dl:>7} {i:60s} {' '.join(t)}")

    print("\n== wisesight_sentiment: real Thai posts by business keyword")
    from datasets import load_dataset
    out_rows = collections.defaultdict(list)
    for split in ("train", "validation", "test"):
        ds = load_dataset("pythainlp/wisesight_sentiment", "wisesight_sentiment", split=split)
        for r in ds:
            t = r["texts"].strip()
            low = t.lower()
            for biz, kws in KEYWORDS.items():
                if any(k in low for k in kws):
                    out_rows[biz].append({"text": t, "label": int(r["category"]), "split": split})
                    break
    for biz, rs in out_rows.items():
        uniq = {r["text"]: r for r in rs}
        with open(os.path.join(args.out, f"wisesight_{biz}.jsonl"), "w", encoding="utf-8") as f:
            for r in uniq.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        lens = [len(r["text"]) for r in uniq.values()]
        print(f"  {biz:10s} {len(uniq):5d} posts, mean {sum(lens) // max(1, len(lens))} chars")
        for r in list(uniq.values())[:4]:
            print("     -", re.sub(r"\s+", " ", r["text"])[:110])


if __name__ == "__main__":
    main()
