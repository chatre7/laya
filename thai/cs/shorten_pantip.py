"""Hand-labelled Pantip posts -> short chat messages, the way a customer types to the company's chat/call center (1-2
sentences), with an OpenAI-compatible LLM (Qwen3-4B on vLLM). The human labels (intent, department, urgency) carry over;
the post's forum framing ("ขอถามผู้รู้", "ใครเคยเจอบ้าง") is dropped. Two variants per post (hurried / polite).

Rows: the 360 eval posts (labels_<biz>.json + pantip_<biz>_review_sample.jsonl) -> set "eval", and the 1,932 training posts
(labels_<biz>_<suffix>.json + pantip_<biz>_<suffix>.jsonl, suffixes extra, extra2) -> set "train".

    python3 thai/cs/shorten_pantip.py --domain thai/data_domain --out thai/data/cs/pantip_short.jsonl
"""
import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BIZ_TH = {"telecom": "ค่ายมือถือ/อินเทอร์เน็ต", "banking": "ธนาคาร", "insurance": "บริษัทประกัน"}
STYLES = ["พิมพ์สั้น ๆ รีบ ๆ แบบแชท ไม่ต้องมีคำลงท้าย", "สุภาพ มีคำลงท้าย ครับ/ค่ะ แต่ยังสั้น"]
PROMPT = ("ข้อความด้านล่างเป็นกระทู้ในเว็บบอร์ด ให้เขียนใหม่เป็นข้อความที่เจ้าของกระทู้จะพิมพ์แชทหาฝ่ายบริการลูกค้าของ{biz}โดยตรง "
          "ยาว 1-2 ประโยค ใช้ภาษาพูด ลักษณะ: {style} "
          "คงเรื่องที่ต้องการและรายละเอียดสำคัญไว้ (ชื่อบริการ ตัวเลข อาการ สิ่งที่เกิดขึ้น) ห้ามเพิ่มข้อมูลใหม่ "
          "ตัดส่วนที่พูดกับคนในเว็บบอร์ดออก (เช่น ขอถามผู้รู้ ใครเคยเจอบ้าง รบกวนพี่ ๆ) "
          "ถ้ากระทู้ไม่ได้ขออะไรจากบริษัทเลย ให้สรุปเนื้อหาเดิมสั้น ๆ ตามที่เป็น ตอบเฉพาะข้อความเดียว ไม่ต้องอธิบาย\n\nกระทู้: {text}")


def chat(url, model, content):
    body = {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0.7, "top_p": 0.95, "max_tokens": 200,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body, ensure_ascii=False).encode(), {"content-type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)["choices"][0]["message"]["content"]
                return re.sub(r"<think>.*?</think>", "", out, flags=re.S).strip().strip('"“”').strip()
        except Exception:  # noqa: BLE001
            time.sleep(1 + attempt)
    return ""


def ok(t):
    thai = sum(1 for ch in t if "฀" <= ch <= "๿")
    alpha = sum(1 for ch in t if ch.isalpha())
    return 6 <= len(t) <= 300 and thai >= 0.5 * max(1, alpha) and "กระทู้" not in t and "\n\n" not in t


def rows(domain):
    out = []
    for biz in ("telecom", "banking", "insurance"):
        for suffix, split in (("", "eval"), ("_extra", "train"), ("_extra2", "train")):
            lab_p = os.path.join(domain, f"labels_{biz}{suffix}.json")
            txt_p = os.path.join(domain, f"pantip_{biz}_review_sample.jsonl" if not suffix else f"pantip_{biz}{suffix}.jsonl")
            if not (os.path.exists(lab_p) and os.path.exists(txt_p)) or os.path.getsize(txt_p) == 0:
                continue
            lab = json.load(open(lab_p, encoding="utf-8"))["labels"]
            for i, line in enumerate(open(txt_p, encoding="utf-8")):
                if str(i) in lab:
                    intent, dept, urg = lab[str(i)]
                    rid = f"{biz}-{i}" if not suffix else f"pantip-{biz}-{suffix[1:]}-{i}"
                    out.append({"id": rid, "set": split, "business": biz, "text": json.loads(line)["text"].strip(),
                                "labels": {"intent": intent, "department": dept, "urgency": int(urg)}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="thai/data_domain")
    ap.add_argument("--out", default="thai/data/cs/pantip_short.jsonl")
    ap.add_argument("--url", default="http://localhost:8012")
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    src = rows(args.domain)
    jobs = [(r, v) for r in src for v in range(len(STYLES))]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        outs = list(ex.map(lambda j: chat(args.url, args.model, PROMPT.format(biz=BIZ_TH[j[0]["business"]], style=STYLES[j[1]],
                                                                               text=" ".join(j[0]["text"].split())[:1500])), jobs))
    kept = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for (r, v), o in zip(jobs, outs):
            o = " ".join(o.split())
            if ok(o):
                kept += 1
                f.write(json.dumps({**{k: r[k] for k in ("id", "set", "business", "labels")}, "variant": v, "text": o}, ensure_ascii=False) + "\n")
    print(f"{len(src)} posts ({sum(r['set'] == 'eval' for r in src)} eval) x {len(STYLES)} -> {kept}/{len(jobs)} kept in {(time.time() - t0) / 60:.1f} min")
    for (r, v), o in list(zip(jobs, outs))[:6]:
        print(f"\n[{r['business']}/{r['labels']['intent']}] {' '.join(r['text'].split())[:120]}\n   -> {o[:150]}")


if __name__ == "__main__":
    main()
